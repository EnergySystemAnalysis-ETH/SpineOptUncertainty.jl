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
    variable_om_costs(m::Model)

Create an expression for unit_flow variable operation costs.
"""
function variable_om_costs(m::Model, t_range)
    return costs_under_risk!(m, variable_om_costs_in_scenario_costs(m, t_range), Val(:expected_value))
end

function variable_om_costs_in_scenario_costs(m::Model, t_range)
    @fetch unit_flow = m.ext[:spineopt].variables
    om_costs = DefaultDict(0.0)
    for (ug, ng, d) in indices(vom_cost)
        for (u, n, d, s, t) in unit_flow_indices(m; unit=ug, node=ng, direction=d, t=t_range)
            om_costs[s] += (
                + unit_flow[u, n, d, s, t]
                * (use_economic_representation(model=m.ext[:spineopt].instance) ?
                   unit_discounted_duration[(unit=u, stochastic_scenario=s, t=t)] : 1
                ) 
                * duration(t)
                * prod(weight(temporal_block=blk) for blk in blocks(t))
                * vom_cost(m; unit=ug, node=ng, direction=d, stochastic_scenario=s, t=t)
            )
        end
    end
    return Dict(om_costs)
end

