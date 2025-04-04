#############################################################################
# Copyright (C) 2017 - 2023  Spine Project
#
# This file is part of SpineOpt.
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
    connection_flow_costs(m::Model)

Create an expression for `connection_flow` costs.
"""
function connection_flow_costs(m::Model, t_range)
    return costs_under_risk!(m, connection_flow_costs_in_scenario_costs(m, t_range), Val(:expected_value))
end

function connection_flow_costs_in_scenario_costs(m::Model, t_range)
    @fetch connection_flow = m.ext[:spineopt].variables
    connection_costs = DefaultDict(0.0)
    for (conn, n, d) in indices(connection_flow_cost)
        for (conn, n, d, s, t) in connection_flow_indices(m; connection=conn, node=n, direction=d, t=t_range)
            connection_costs[s] += (
                connection_flow[conn, n, d, s, t]
                * duration(t)
                * (use_economic_representation(model=m.ext[:spineopt].instance) ?
                   connection_discounted_duration[(connection=conn, stochastic_scenario=s, t=t)] : 1
                )
                * prod(weight(temporal_block=blk) for blk in blocks(t))
                * connection_flow_cost(m; connection=conn, node=n, direction=d, stochastic_scenario=s, t=t)
            )
        end
    end
    return Dict(connection_costs)
end