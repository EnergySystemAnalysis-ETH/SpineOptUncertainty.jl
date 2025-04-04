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
    units_on_costs(m::Model)

Create an expression for units_on cost.
"""
function units_on_costs(m::Model, t_range)
    return costs_under_risk!(m, units_on_costs_in_scenario_costs(m, t_range), Val(:expected_value))
end

function units_on_costs_in_scenario_costs(m::Model, t_range)
    @fetch units_on = m.ext[:spineopt].variables
    units_on_costs = DefaultDict(0.0)
    for (u, s, t) in units_on_indices(m; unit=indices(units_on_cost), t=t_range)
        units_on_costs[s] += (
            + units_on[u, s, t]
            * (use_economic_representation(model=m.ext[:spineopt].instance) ?
               unit_discounted_duration[(unit=u, stochastic_scenario=s, t=t)] : 1
            ) 
            * duration(t)
            * units_on_cost(m; unit=u, stochastic_scenario=s, t=t)
            * prod(weight(temporal_block=blk) for blk in blocks(t))
        )
    end
    return Dict(units_on_costs)
end
