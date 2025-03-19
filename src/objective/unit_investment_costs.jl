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
    unit_investment_costs(m::Model)

Create and expression for unit investment costs.
"""
function unit_investment_costs(m::Model, t_range)
    return costs_under_risk!(m, unit_investment_costs_in_scenario_costs(m, t_range), Val(:expected_value))
end

function _unit_weight_for_economic_representation(m; u, s, t)
    if use_economic_representation(model=m.ext[:spineopt].instance)
        return (1- unit_salvage_fraction[(unit=u, stochastic_scenario=s, t=t)]) * 
                unit_tech_discount_factor[(unit=u, stochastic_scenario=s, t=t)] * 
                unit_conversion_to_discounted_annuities[(unit=u, stochastic_scenario=s, t=t)]
    else
        return 1
    end
end


function unit_investment_costs_in_scenario_costs(m::Model, t_range)
    @fetch units_invested = m.ext[:spineopt].variables
    unit = indices(unit_investment_cost)
    unit_investment_costs = DefaultDict(0.0)
    for (u, s, t) in units_invested_available_indices(m; unit=unit, t=t_range)
        unit_investment_costs[s] += (
            + units_invested[u, s, t]
            * _unit_weight_for_economic_representation(m; u, s, t)
            * unit_investment_cost(m; unit=u, stochastic_scenario=s, t=t)
            * prod(weight(temporal_block=blk) for blk in blocks(t))
        )
    end
    return Dict(unit_investment_costs)
end
