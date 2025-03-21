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


scenario_probability(m::Model) = scen -> any_stochastic_scenario_weight(m; stochastic_scenario=scen)

function costs_under_risk!(m::Model, scenario_costs::Dict, ::Val{:expected_value})
    return expected_value(scenario_costs, scenario_probability(m))
end

function costs_under_risk!(m::Model, scenario_costs::Dict, ::Val{:cvar})
    beta = cvar_percentage(model=m.ext[:spineopt].instance)
    return cvar(m, beta, scenario_costs, scenario_probability(m)) 
end

function costs_under_risk!(m::Model, scenario_costs::Dict, ::Val{:markowitz})
    lambda = dispersion_to_mean_cost_weight(model=m.ext[:spineopt].instance)
    dispersion_type = Val(dispersion_measure(model=m.ext[:spineopt].instance))
    return markowitz_model(m::Model, lambda, scenario_costs, scenario_probability(m), dispersion_type)
end

function positive_part_of_lp_term(m::Model, term, auxilary_var_name::String="")
    d = @variable(m, lower_bound=0, base_name=auxilary_var_name)
    @constraint(m, d >= term)
    return d
end

expected_value(scenario_costs, probability::Function) = sum(cost * probability(scen) for (scen, cost) in scenario_costs; init=0)

function cvar(m::Model, beta::Float64, scenario_costs::Dict, scenario_probability::Function)
    !(0 < beta <= 1) && throw(DomainError(beta, "parameter not in the domain 0 < beta <= 1"))
    @variable(m, v) # TODO: save the v
    p = scenario_probability
    return v + 1/beta * sum(p(scen) * positive_part_of_lp_term(m, cost - v) for (scen, cost) in scenario_costs; init=0)
end

function markowitz_model(m::Model, lambda::Float64, scenario_costs::Dict, scenario_probability::Function, dispersion_type::Val)
    !(0 < lambda <= 1) && throw(DomainError(lambda, "parameter not in the domain 0 < lambda <= 1"))
    mu = expected_value(scenario_costs, scenario_probability)
    delta = dispersion_metric(m, mu, scenario_costs, scenario_probability, dispersion_type)
    return mu + lambda * delta
end

function dispersion_metric(m::Model, mu, scenario_costs::Dict, probability::Function, ::Val{:max_semideviation})
    d = @variable(m, lower_bound=0)
    for (scen, cost) in scenario_costs
        @constraint(m, d >= cost - mu)
    end
    return d
end

function dispersion_metric(m::Model, mu, scenario_costs::Dict, probability::Function, ::Val{:avg_semideviation})
    return sum(probability(scen) * positive_part_of_lp_term(m, cost - mu) for (scen, cost) in scenario_costs; init=0)
end

function dispersion_metric(m::Model, mu, scenario_costs::Dict, probability::Function, ::Val{:avg_gini_difference})
    return sum(
        positive_part_of_lp_term(m, cost1 - cost2) * probability(scen1) * probability(scen2)
        for (scen1, cost1) in scenario_costs for (scen2, cost2) in scenario_costs if scen1 != scen2
        ; init=0
    )
end