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
# TODO works only with trees
path_probability(m::Model) = path -> any_stochastic_scenario_weight(m; stochastic_scenario=path[end])

generate_path_costs(scenario_costs, stochastic_paths) = Dict(path => sum(get(scenario_costs, scen, 0) for scen in path) for path in stochastic_paths)


"""
    Scalarizes the stochastic goal function with expected value operator.

    # Arguments:
    - m - JuMP model
    - scenario_dict - a dictionary with scenarios as keys and their respective cost functions as values
"""
function costs_under_risk!(m::Model, scenario_costs::Dict, ::Val{:expected_value})
    #return expected_value(scenario_costs, scenario_probability(m))
    path_costs = generate_path_costs(scenario_costs, m.ext[:spineopt].stochastic_structure[:full_stochastic_paths])
    return expected_value(path_costs, path_probability(m))
end

"""
    Scalarizes the stochastic goal function with CVAR operator.

    # Arguments:
    - m - JuMP model
    - scenario_dict - a dictionary with scenarios as keys and their respective cost functions as values
"""
function costs_under_risk!(m::Model, scenario_costs::Dict, ::Val{:cvar})
    beta = cvar_percentage(model=m.ext[:spineopt].instance)
    path_costs = generate_path_costs(scenario_costs, m.ext[:spineopt].stochastic_structure[:full_stochastic_paths])
    return cvar(m, beta, path_costs, path_probability(m)) 
end

"""
    Scalarizes the stochastic goal function with Markowitz model formulation.

    # Arguments:
    - m - JuMP model
    - scenario_costs - a dictionary with scenarios as keys and their respective cost functions as values
"""
function costs_under_risk!(m::Model, scenario_costs::Dict, ::Val{:markowitz})
    lambda = dispersion_to_mean_cost_weight(model=m.ext[:spineopt].instance)
    dispersion_type = Val(dispersion_measure(model=m.ext[:spineopt].instance))
    path_costs = generate_path_costs(scenario_costs, m.ext[:spineopt].stochastic_structure[:full_stochastic_paths])
    return markowitz_model(m::Model, lambda, path_costs, path_probability(m), dispersion_type)
end

"""
    Returns expected value for given goal functions per scenarios and their appropriate probabilities

    # Arguments:
    - scenario_costs - a dictionary with scenarios as keys and their respective cost functions as values
    - probability - a function that for evey scenario returns its probability 
"""
expected_value(scenario_costs, probability::Function) = sum(cost * probability(scen) for (scen, cost) in scenario_costs; init=0)

"""
    Returns CVAR LP formulation for given goal functions per scenario paths and their appropriate probabilities.
    It results in optimization of beta worst cases (highest costs).

    # Arguments:
    - m - JuMP model
    - beta - percentage of worst cases optimized by the CVAR
    - path_costs - a dictionary with paths as keys and their respective cost functions as values
    - path_probability - a function that for evey path returns its probability 
"""
function cvar(m::Model, beta, path_costs::Dict, path_probability::Function)
    !(0 < beta <= 1) && throw(DomainError(beta, "parameter not in the domain 0 < beta <= 1"))
    @variable(m, v)
    return v + 1/beta * sum(path_probability(scen) * positive_part_of_lp_term(m, cost - v) for (scen, cost) in path_costs; init=0)
end

"""
    Returns Markowitz LP formulation for given goal functions per scenario paths and their appropriate probabilities.
    It is a scalarization of problem of both cost and dispersion minimization.

    # Arguments:
    - m - JuMP model
    - lambda - factor of tradeoff between dispersion and expected value, the lower the more attention is given to expectd value
    - path_costs - a dictionary with paths as keys and their respective cost functions as values
    - path_probability - a function that for evey scenario returns its probability 
"""
function markowitz_model(m::Model, lambda, path_costs::Dict, path_probability::Function, dispersion_type::Val)
    !(0 < lambda < 1) && throw(DomainError(lambda, "parameter not in the domain 0 < lambda < 1"))
    mu = expected_value(path_costs, path_probability)
    delta = dispersion_metric(m, path_costs, path_probability, dispersion_type)
    return mu + lambda * delta
end

"""
    Returns an LP formulation for minimization of only positive part of an expression.
    
    # Arguments:
    - m - JuMP model
    - term - expression for which positive part is sought
    - auxilary_var_name - a parameter to name the resulting auxilary variable
"""
function positive_part_of_lp_term(m::Model, term, auxilary_var_name::String="")
    d = @variable(m, lower_bound=0, base_name=auxilary_var_name)
    @constraint(m, d >= term)
    return d
end

"""
    Return an LP formulation for minimization of a (upper) semideviation.
    Here we optimize over only the terms were the costs rise above the average cost.

    # Arguments:
    - m - JuMP model
    - cost - an expression indicating a single scenario cost
    - mu - an expression describing average cost over all scenarios 
"""
semideviation(m::Model, cost, mu) = positive_part_of_lp_term(m, cost - mu)

"""
    Returns an LP formulation for maximal semideviation metric.
    We want to minimize the worst upper semideviation, resulting from the highest cost above the expected cost.
    
    # Arguments:
    - m - JuMP model
    - path_costs - a dictionary with scenario paths as keys and their respective cost functions as values
    - probability - a function that for every scenario path returns its probability 
"""
function dispersion_metric(m::Model, path_costs::Dict, probability::Function, ::Val{:max_semideviation})
    mu = expected_value(path_costs, probability)
    d = @variable(m, lower_bound=0)
    for (scen, cost) in path_costs
        @constraint(m, d >= semideviation(m, cost, mu))
    end
    return d
end

"""
    Returns an LP formulation for average semideviation metric.
    We want to minimize the average upper semideviation - the average value of scenario cost deviations above the average value.
    
    # Arguments:
    - m - JuMP model
    - path_costs - a dictionary with paths as keys and their respective cost functions as values
    - probability - a function that for every path returns its probability 
"""
function dispersion_metric(m::Model, path_costs::Dict, probability::Function, ::Val{:avg_semideviation})
    mu = expected_value(path_costs, probability)
    scenario_semideviations = Dict(scen => semideviation(m, cost, mu) for (scen, cost) in path_costs)
    return expected_value(scenario_semideviations, probability)
end

"""
    Returns an LP formulation for average Gini difference.
    We want to minimize the average differences between scenario path cost, thus to achieve as clustered results as possible.
    
    # Arguments:
    - m - JuMP model
    - path_costs - a dictionary with scenario paths as keys and their respective cost functions as values
    - probability - a function that for every scenario oath returns its probability 
"""
function dispersion_metric(m::Model, path_costs::Dict, probability::Function, ::Val{:avg_gini_difference})
    return sum(
        positive_part_of_lp_term(m, cost1 - cost2) * probability(scen1) * probability(scen2)
        for (scen1, cost1) in path_costs for (scen2, cost2) in path_costs if scen1 != scen2
        ; init=0
    )
end