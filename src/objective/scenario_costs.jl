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

const scenario_cost_parts = [
    :connection_flow_costs_in_scenario_costs,
    :connection_investment_costs_in_scenario_costs,
    :fixed_om_costs_in_scenario_costs,
    :fuel_costs_in_scenario_costs,
    :min_capacity_margin_penalties_in_scenario_costs,
    :objective_penalties_in_scenario_costs,
    :renewable_curtailment_costs_in_scenario_costs,
    :res_proc_costs_in_scenario_costs,
    :shut_down_costs_in_scenario_costs,
    :start_up_costs_in_scenario_costs,
    :storage_investment_costs_in_scenario_costs,
    :taxes_in_scenario_costs,
    :unit_investment_costs_in_scenario_costs,
    :units_on_costs_in_scenario_costs,
    :variable_om_costs_in_scenario_costs
]

create_scenario_costs(m, t_range) = mergewith(+, (getproperty(SpineOpt, cost_part)(m, t_range) for cost_part in scenario_cost_parts)...)
create_scenario_costs(m, in_window, beyond_window) = mergewith(+, create_scenario_costs(m, in_window), create_scenario_costs(m, beyond_window))