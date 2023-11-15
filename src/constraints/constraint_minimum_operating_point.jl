#############################################################################
# Copyright (C) 2017 - 2023  Spine Project
#
# This file is part of SpineOpt.
#
# SpineOpt is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SpineOpt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################

"""
    add_constraint_minimum_operating_point!(m::Model)

Limit the maximum in/out `unit_flow` of a `unit` if the parameters
`unit_capacity`, `number_of_units`, `unit_conv_cap_to_flow`, and `unit_availability_factor` exist.
"""
function add_constraint_minimum_operating_point!(m::Model)
    @fetch unit_flow, units_on, nonspin_units_started_up, nonspin_units_shut_down = m.ext[:spineopt].variables
    t0 = _analysis_time(m)
    m.ext[:spineopt].constraints[:minimum_operating_point] = Dict(
        (unit=u, node=ng, direction=d, stochastic_path=s, t=t) => @constraint(
            m,
            + expr_sum(
                + unit_flow[u, n, d, s, t_short] * duration(t_short)
                for (u, n, d, s, t_short) in unit_flow_indices(
                    m; unit=u, node=ng, direction=d, stochastic_scenario=s, t=t_in_t(m, t_long=t)
                )
                if !is_reserve_node(node=n);
                init=0,
            )
            - expr_sum(
                + unit_flow[u, n, d, s, t_short] * duration(t_short)
                for (u, n, d, s, t_short) in unit_flow_indices(
                    m; unit=u, node=ng, direction=d, stochastic_scenario=s, t=t_in_t(m, t_long=t)
                )
                if is_reserve_node(node=n) && _switch(d; to_node=downward_reserve, from_node=upward_reserve)(node=n);
                init=0,
            )
            >=
            + expr_sum(
                (
                    + units_on[u, s, t_over]
                    - expr_sum(
                        _switch(
                            d; from_node=nonspin_units_started_up, to_node=nonspin_units_shut_down
                        )[u, n, s, t]
                        for (u, n, s, t) in _switch(
                            d; from_node=nonspin_units_started_up_indices, to_node=nonspin_units_shut_down_indices
                        )(m; unit=u, node=ng, stochastic_scenario=s, t=t_over);
                        init=0
                    )
                )
                * min(duration(t), duration(t_over))
                * minimum_operating_point[(unit=u, node=ng, direction=d, stochastic_scenario=s, analysis_time=t0, t=t)]
                * unit_capacity[(unit=u, node=ng, direction=d, stochastic_scenario=s, analysis_time=t0, t=t)]
                * unit_conv_cap_to_flow[(unit=u, node=ng, direction=d, stochastic_scenario=s, analysis_time=t0, t=t)]
                for (u, s, t_over) in units_on_indices(m; unit=u, stochastic_scenario=s, t=t_overlaps_t(m; t=t));
                init=0,
            )
        )
        for (u, ng, d, s, t) in constraint_minimum_operating_point_indices(m)
    )
end

function constraint_minimum_operating_point_indices(m::Model)
    unique(
        (unit=u, node=ng, direction=d, stochastic_path=path, t=t)
        for (u, ng, d) in indices(minimum_operating_point)
        for (t, path) in t_lowest_resolution_path(
            m, unit_flow_indices(m; unit=u, node=ng, direction=d), units_on_indices(m; unit=u)
        )
    )
end

"""
    constraint_minimum_operating_point_indices_filtered(m::Model; filtering_options...)

Form the stochastic indexing Array for the `:minimum_operating_point` constraint.

Uses stochastic path indices due to potentially different stochastic structures between
`unit_flow` and `units_on` variables. Keyword arguments can be used to filter the resulting Array.
"""
function constraint_minimum_operating_point_indices_filtered(
    m::Model;
    unit=anything,
    node=anything,
    direction=anything,
    stochastic_path=anything,
    t=anything,
)
    f(ind) = _index_in(ind; unit=unit, node=node, direction=direction, stochastic_path=stochastic_path, t=t)
    filter(f, constraint_minimum_operating_point_indices(m))
end
