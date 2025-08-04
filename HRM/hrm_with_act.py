from __future__ import annotations

from collections import namedtuple
from contextlib import nullcontext
from random import randrange, random

import torch
import torch.nn.functional as F
from torch import Tensor, tensor, is_tensor, arange, cat, stack, maximum
from torch.nn import Embedding, Linear, Sequential, Module, ModuleList

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from x_transformers import Encoder, Decoder, RMSNorm

# constants

Losses = namedtuple('Losses', ['main', 'act'])

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def last(arr):
    return arr[-1]

def satisfy_prob(prob):
    return random() < prob

def is_empty(t: Tensor):
    return t.numel() == 0

def divisible_by(num, den):
    return (num % den) == 0

# combining hiddens across hierarchies

class CombineHiddens(Module):
    def __init__(
        self,
        dim,
        num_hiddens_to_concat
    ):
        super().__init__()
        self.num_hiddens_to_concat = num_hiddens_to_concat

        self.norms = ModuleList([RMSNorm(dim) for _ in range(num_hiddens_to_concat)])

        self.to_combined = Linear(dim * self.num_hiddens_to_concat, dim, bias = False)

    def forward(
        self,
        hiddens: list[Tensor],
        hierarchy_index
    ):
        hiddens_to_concat = hiddens[hierarchy_index:]

        assert len(hiddens_to_concat) == self.num_hiddens_to_concat

        normed = tuple(norm(t) for norm, t in zip(self.norms, hiddens))

        concatted = cat(normed, dim = -1)

        return self.to_combined(concatted)

# modules

class HRM(Module):
    def __init__(
        self,
        networks: list[Module | dict],
        *,
        dim,
        num_tokens,
        reasoning_steps = 2,                          # N in the paper - the number of forward evals for the last network (highest hierarchy) above
        relative_period: int | tuple[int, ...] = 2,   # the relative period for each network evaluation call to the one just previous - in the paper, they do 2 networks with a period of 2
        causal = False,
        min_reasoning_steps_epsilon_prob = 0.5,            # they stochastically choose the minimum segment from 2 .. max with this probability, and 1 step the rest of the time
        max_reasoning_steps = 10,
        act_loss_weight = 1.,
        discount_factor = 1.,
        ignore_index = -1,
    ):
        super().__init__()
        attn_layers_klass = Encoder if not causal else Decoder
        self.causal = causal

        # input

        self.to_input_embed = Embedding(num_tokens, dim)

        # allow for any number of hierarchical modules

        # order in hierarchy should be from low to high

        self.networks = ModuleList()

        for network in networks:
            if isinstance(network, dict):
                network = attn_layers_klass(**network)

            self.networks.append(network)

        self.num_networks = len(self.networks)
        assert self.num_networks > 0

        # setup how frequent each network is called
        # the first network (lowest in the hierarchy) should be called every iteration

        num_higher_networks = self.num_networks - 1

        if not isinstance(relative_period, tuple):
            relative_period = (relative_period,) * num_higher_networks

        # implied that first network is called always

        if len(relative_period) == (self.num_networks - 1):
            relative_period = (1, *relative_period)

        # for the paper, they did (low: 1, high: 2) -

        assert len(relative_period) == self.num_networks and relative_period[0] == 1

        self.evaluate_networks_at = tensor(relative_period).cumprod(dim = -1).tolist()

        self.reasoning_steps = reasoning_steps
        self.lowest_steps_per_reasoning_step = last(self.evaluate_networks_at)

        # combining hiddens

        self.hidden_combiners = ModuleList([CombineHiddens(dim, self.num_networks + 1 - network_index) for network_index in range(self.num_networks)])

        # output

        self.to_logits = Linear(dim, num_tokens, bias = False)

        # Q(continue|halt) for their adaptive computation time setup

        self.discount_factor = discount_factor

        self.act_loss_weight = act_loss_weight

        self.min_reasoning_steps_epsilon_prob = min_reasoning_steps_epsilon_prob
        self.max_reasoning_steps = max_reasoning_steps

        self.to_q_halt_continue = Sequential(
            Reduce('b n d -> b d', 'mean'),
            RMSNorm(dim),
            Linear(dim, 2, bias = False),
            Rearrange('... halt_continue -> halt_continue ...')
        )

        # loss related

        self.ignore_index = ignore_index

    def forward(
        self,
        seq,
        hiddens: tuple[Tensor, ...] | None = None,
        *,
        labels = None,
        compute_loss_across_reasoning_steps = False,
        detach_hiddens = True,
        one_step_grad = True,
        max_reasoning_steps = None,
        adaptive_compute = None,
        return_autoreg_loss = False
    ):
        batch, device = seq.shape[0], seq.device

        highest_hidden_index = self.num_networks - 1

        if return_autoreg_loss:
            assert self.causal
            seq, labels = seq[:, :-1], seq[:, 1:]

        return_loss = exists(labels)

        # ACT related variables

        adaptive_compute = default(adaptive_compute, not return_loss)

        assert not (adaptive_compute and return_loss)

        max_reasoning_steps = default(max_reasoning_steps, self.max_reasoning_steps)

        # detach all previous hiddens cmoing in

        if exists(hiddens) and detach_hiddens:
            hiddens = tuple(h.detach() for h in hiddens)

        # seq to input tokens

        tokens = self.to_input_embed(seq)

        # handle hidden default to zeros, add learned hiddens some other day, should make little difference

        if not exists(hiddens):
            hiddens = torch.zeros_like(tokens)
            hiddens = repeat(hiddens, '... -> num_networks ...', num_networks = self.num_networks)

        assert len(hiddens) == self.num_networks

        # hiddens to a dictionary, avoid some inplace error when updating hidden
 
        hiddens = {index: hidden for index, hidden in enumerate(hiddens)}

        # maybe 1-step

        min_reasoning_steps = self.max_reasoning_steps

        if self.training:
            min_reasoning_steps = randrange(2, max_reasoning_steps + 1) if satisfy_prob(self.min_reasoning_steps_epsilon_prob) else 1

        if not adaptive_compute:
            # variables for storing the predicted q_halt_continue and hiddens for learning during training

            highest_hiddens = []
            pred_q_halt_continues = []
        else:

            # variables for ACT during inference
            # we will select out the batch elements that exit early, keeping track of the indices, then sort it back at the end

            current_indices = arange(batch, device = device)

            completed_indices = current_indices[0:0] # indices will be added from `current_indices` to this if they exit early

            reasoning_steps_for_samples = current_indices[0:0] # for returning the exit layer per batch sample Int['b']

            # keep track of the hiddens for each batch sample at its exit layer

            exit_hiddens = {index: hidden[0:0] for index, hidden in hiddens.items()}

        # going through the networks

        total_steps = max_reasoning_steps * self.lowest_steps_per_reasoning_step

        for index in range(total_steps):

            iteration = index + 1

            is_reasoning_step_boundary = divisible_by(iteration, self.lowest_steps_per_reasoning_step)
            num_reasoning_steps = iteration // self.lowest_steps_per_reasoning_step

            # evaluate all networks depending on their period

            is_last_step = index == (total_steps - 1)
            context = torch.no_grad if one_step_grad and not is_last_step else nullcontext

            with context():
                for network_index, (network, hidden_combine, evaluate_network_at) in enumerate(zip(self.networks, self.hidden_combiners, self.evaluate_networks_at)):

                    if not divisible_by(iteration, evaluate_network_at):
                        continue

                    all_hiddens = (
                        tokens,
                        *hiddens.values()
                    )

                    # combine with mean pool for now

                    combined_input = hidden_combine(all_hiddens, network_index)

                    # forward

                    next_hidden = network(combined_input)

                    # store hiddens at appropriate hierarchy, low to highest

                    hiddens[network_index] = next_hidden

            # adaptive computation time

            if not is_reasoning_step_boundary:
                continue

            highest_hidden = hiddens[highest_hidden_index]

            q_halt_continue = self.to_q_halt_continue(highest_hidden).sigmoid()

            # not training

            q_halt, q_continue = q_halt_continue

            if adaptive_compute:
                should_halt_at_step = q_halt > q_continue

                halted_indices = current_indices[should_halt_at_step]

                completed_indices = cat((completed_indices, halted_indices))

                curr_reasoning_step = torch.ones_like(halted_indices).long() * num_reasoning_steps
                reasoning_steps_for_samples = cat((reasoning_steps_for_samples, curr_reasoning_step))

                current_indices = current_indices[~should_halt_at_step]

                tokens = tokens[~should_halt_at_step]

                # update hiddens for exited, in order of exit, to be sorted back later

                exit_hiddens = {index: cat((exit_hidden, hiddens[index][should_halt_at_step])) for index, exit_hidden in exit_hiddens.items()}

                # update hiddens to only have those non-exited

                hiddens = {index: hidden[~should_halt_at_step] for index, hidden in hiddens.items()}

                # break out of loop if all exited

                if is_empty(current_indices):
                    break

            if return_loss:

                highest_hidden = hiddens[highest_hidden_index]

                highest_hiddens.append(highest_hidden)

                pred_q_halt_continues.append(q_halt_continue)

        # take care of ACT vs without

        if adaptive_compute:

            # append remaining unexited hiddens

            exit_hiddens = {index: cat((exit_hidden, hiddens[index])) for index, exit_hidden in exit_hiddens.items()}

            exited_indices_order = cat((completed_indices, current_indices))

            # sorting back to original batch indices order

            indices_to_orig_batch_order = exited_indices_order.argsort(dim = -1)

            reasoning_steps_for_samples = F.pad(reasoning_steps_for_samples, (0, batch - reasoning_steps_for_samples.shape[0]), value = -1) # should be -1 for the remaining unexited samples, or maybe max reasoning steps + 1

            reasoning_steps_for_samples = reasoning_steps_for_samples[indices_to_orig_batch_order]

            exit_hiddens = {index: hidden[indices_to_orig_batch_order] for index, hidden in exit_hiddens.items()}

            highest_hidden = exit_hiddens[highest_hidden_index]

            # output hiddens will be the hidden at exit

            hiddens_out = list(exit_hiddens.values())
        else:
            hiddens_out = list(hiddens.values())
            highest_hidden = hiddens[highest_hidden_index]

        if not return_loss:
            # to output prediction, using the hiddens from the highest hierarchy

            logits = self.to_logits(highest_hidden)

            if not adaptive_compute:
                return logits, hiddens_out

            return logits, hiddens_out, reasoning_steps_for_samples

        # get main loss

        highest_hiddens = stack(highest_hiddens) # (l b n d)

        if not compute_loss_across_reasoning_steps:
            logits = self.to_logits(highest_hiddens[-1])

            main_pred_loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                labels,
                ignore_index = self.ignore_index
            )

        else:
            all_logits = self.to_logits(highest_hiddens)
            num_layers = all_logits.shape[0]

            main_pred_loss = F.cross_entropy(
                rearrange(all_logits, 'l b n c -> b c n l'),
                repeat(labels, 'b n -> b n l', l = num_layers),
                ignore_index = self.ignore_index
            )

        # compute the act loss

        q_halts, q_continues = rearrange(pred_q_halt_continues, 'l halt_continue b -> halt_continue l b')

        # q halt loss is simply on whether the prediction is correct or not

        with torch.no_grad():
            all_logits = self.to_logits(highest_hiddens)
            is_correct = (all_logits.argmax(dim = -1) == labels).all(dim = -1)

        q_halt_losses = F.binary_cross_entropy(
            q_halts,
            is_correct.float(),
            reduction = 'none'
        )

        # q continue is learned using bellman's on max(q_halt, q_continue) of next reasoning step

        mask_value = -torch.finfo(q_continues.dtype).max

        q_max_halt_continue = maximum(
            q_halts[1:-1],
            q_continues[1:-1]
        )

        q_continue_target = cat((q_max_halt_continue, q_halts[-1:])) # last step the q_continue target is just q_halt

        q_continue_losses = F.binary_cross_entropy(
            q_continues[:-1],
            q_continue_target * self.discount_factor, # they use a discount factor of 1., don't understand why yet
            reduction = 'none'
        )

        # average loss for learning the q values

        act_loss = cat((q_halt_losses, q_continue_losses), dim = 0).mean()

        # total loss + loss breakdown

        total_loss = (
            main_pred_loss +
            act_loss * self.act_loss_weight
        )

        loss_breakdown = Losses(main_pred_loss, act_loss)

        return total_loss, hiddens_out, (logits, loss_breakdown)
