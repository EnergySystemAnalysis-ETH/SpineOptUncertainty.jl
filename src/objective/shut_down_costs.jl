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
    shut_down_costs(m::Model)

Create an expression for unit shutdown costs.
"""
function shut_down_costs(m::Model, t_range)
    return costs_under_risk!(m, shut_down_costs_in_scenario_costs(m, t_range), Val(:expected_value))
end

function shut_down_costs_in_scenario_costs(m::Model, t_range)
    @fetch units_shut_down = m.ext[:spineopt].variables
    unit_shut_down_costs = DefaultDict(0.0)
    for (u, s, t) in units_on_indices(m; unit=indices(shut_down_cost), t=t_range)
        unit_shut_down_costs[s] += (
            + units_shut_down[u, s, t]
            * shut_down_cost(m; unit=u, stochastic_scenario=s, t=t)
            * (use_economic_representation(model=m.ext[:spineopt].instance) ?
               unit_discounted_duration[(unit=u, stochastic_scenario=s, t=t)] : 1
            ) 
            * prod(weight(temporal_block=blk) for blk in blocks(t))
        )
    end
    return Dict(unit_shut_down_costs)
end
