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
    storage_investment_costs(m::Model)

Create and expression for storage investment costs.
"""
function storage_investment_costs(m::Model, t_range)
    return costs_under_risk!(m, storage_investment_costs_in_scenario_costs(m, t_range), Val(:expected_value))
end
    
function _storage_weight_for_economic_representation(m; n, s, t)
    if use_economic_representation(model=m.ext[:spineopt].instance)
        return (1- storage_salvage_fraction[(node=n, stochastic_scenario=s, t=t)]) * 
                storage_tech_discount_factor[(node=n, stochastic_scenario=s, t=t)] * 
                storage_conversion_to_discounted_annuities[(node=n, stochastic_scenario=s, t=t)]
    else
        return 1
    end
end

function storage_investment_costs_in_scenario_costs(m::Model, t_range)
    @fetch storages_invested = m.ext[:spineopt].variables
    node = indices(storage_investment_cost)
    storage_investment_costs = DefaultDict(0.0)
    for (n, s, t) in storages_invested_available_indices(m; node=node, t=t_range)
        storage_investment_costs[s] += (
            + storages_invested[n, s, t]
            * _storage_weight_for_economic_representation(m; n, s, t)
            * storage_investment_cost(m; node=n, stochastic_scenario=s, t=t)
            * prod(weight(temporal_block=blk) for blk in blocks(t))
        )
    end
    return Dict(storage_investment_costs)
end