#############################################################################
# Copyright (C) 2017 - 2018  Spine Project
#
# This file is part of Spine Model.
#
# Spine Model is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spine Model is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################

"""
    constraint_max_start_up_ramp_indices()

Form the stochastic index set for the `:max_start_up_ramp` constraint.
    
Uses stochastic path indices due to potentially different stochastic scenarios between `t_after` and `t_before`.
"""
function constraint_max_start_up_ramp_indices()
    unique(
        (unit=u, node=ng, direction=d, stochastic_path=path, t=t)
        for (u, ng, d) in indices(max_startup_ramp)
        for t in t_lowest_resolution(time_slice(temporal_block=node__temporal_block(node=members(ng))))
        # How to deal with groups correctly?
        for path in active_stochastic_paths(
            unique(
                ind.stochastic_scenario
                for ind in Iterators.flatten(
                    (units_on_indices(unit=u, t=t), start_up_unit_flow_indices(unit=u, node=ng, direction=d, t=t))
                )  # Current `units_on` and `units_available`, plus `units_shut_down` during past time slices
            )
        )
    )
end

"""
    add_constraint_max_start_up_ramp!(m::Model)

Limit the maximum ramp at the start up of a unit. For reserves the max non-spinning
reserve ramp can be defined here.
"""
# TODO: Good to go for first try; make sure capacities are well defined
function add_constraint_max_start_up_ramp!(m::Model)
    @fetch units_started_up, start_up_unit_flow = m.ext[:variables]
    m.ext[:constraints][:max_start_up_ramp] = Dict(
        (u, ng, d, s, t) => @constraint(
            m,
            + sum(
                start_up_unit_flow[u, n, d, s, t]
                for (u, n, d, s, t) in start_up_unit_flow_indices(
                    unit=u, node=ng, direction=d, stochastic_scenario=s, t=t_in_t(t_long=t)
                )
            )
            <=
            + sum(
                units_started_up[u, s, t]
                for (u, s, t) in units_on_indices(unit=u, stochastic_scenario=s, t=t_overlaps_t(t))
            )
            * max_startup_ramp[(unit=u, node=ng, direction=d)]
            * unit_conv_cap_to_flow[(unit=u, node=ng, direction=d, t=t)]
            * unit_capacity[(unit=u, node=ng, direction=d, t=t)]
        )
        for (u, ng, d, s, t) in constraint_max_start_up_ramp_indices()
    )
end
