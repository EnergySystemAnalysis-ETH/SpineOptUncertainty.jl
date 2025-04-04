#############################################################################
# Copyright (C) 2017 - 2018  Spine Project
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

using SpineOpt:
    expected_value,
    positive_part_of_lp_term,
    semideviation,
    dispersion_metric,
    cvar,
    markowitz_model

using JuMP
using HiGHS

function test_expected_value()
    @testset "test expected value" begin
        @testset "empty scenarios" begin
            scenario_costs = Dict()
            scenario_probabilities = Dict(:a => 1.0)
            prob = (i) -> scenario_probabilities[i]
            @test expected_value(scenario_costs, prob) ==  0.0
        end
        @testset "all scenarios with zero values" begin
            scenario_costs = Dict(:a => 0.0, :b => 0.0)
            scenario_probabilities = Dict(:a => 0.5, :b => 0.5)
            prob = (i) -> scenario_probabilities[i]
            @test expected_value(scenario_costs, prob) ==  0.0
        end
        @testset "expected value of floats" begin
            scenario_costs = Dict(:a => 1.0, :b => 2.0)
            scenario_probabilities = Dict(:a => 0.7, :b => 0.3)
            prob = (i) -> scenario_probabilities[i]
            @test isapprox(expected_value(scenario_costs, prob), 1.3)
        end
        @testset "expected value with variables" begin
            m = Model()
            @variable(m, x[1:2])
            scenario_costs = Dict(
                :a => x[1] + 2x[2] + 1, 
                :b => 0.5x[2],
                :c => 0
            )
            scenario_probabilities = Dict(:a => 0.5, :b => 0.25, :c => 0.25)
            prob = (i) -> scenario_probabilities[i]
            @test expected_value(scenario_costs, prob) == 0.5x[1] + 1.125x[2] + 0.5
        end
    end
end

function test_positive_part()
    @testset "positive part of lp term" begin
        @testset "positive part of a float terms" begin
            @testset "positive part of negative term" begin
                m = Model(HiGHS.Optimizer)
                d = positive_part_of_lp_term(m, -5)
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.
            end
            @testset "positive part of zero term" begin
                m = Model(HiGHS.Optimizer)
                d = positive_part_of_lp_term(m, 0)
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.
            end
            @testset "positive part of positive term" begin
                @testset "positive part of zero term" begin
                    m = Model(HiGHS.Optimizer)
                    d = positive_part_of_lp_term(m, 5)
                    @objective(m, Min, d)
                    set_silent(m)
                    optimize!(m)
                    @test value(d) == 5.
                end 
            end
        end
        @testset "positive part of a expression terms" begin
            @testset "positive part of negative term" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, -2 <= x)
                d = positive_part_of_lp_term(m, x)
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.
            end
            @testset "positive part of zero term" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, 0 <= x)
                d = positive_part_of_lp_term(m, x)
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.
            end
            @testset "positive part of positive term" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, 1 <= x)
                d = positive_part_of_lp_term(m, x)
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 1.
            end
        end
    end
    @testset "auxilary_var_name" begin
        @testset "no name given" begin
            m = Model(HiGHS.Optimizer)
            d = positive_part_of_lp_term(m, 0)
            @test name(d) == ""
        end
        @testset "name given" begin
            m = Model(HiGHS.Optimizer)
            d = positive_part_of_lp_term(m, 0, "abc")
            @test name(d) == "abc"
        end
        
    end
end

function test_semideviation()
    @testset "semideviation" begin
        @testset "semideviation with floats" begin
            @testset "no deviation" begin
                m = Model(HiGHS.Optimizer)
                d = semideviation(m, 1.0, 1.0)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.0
            end
            @testset "negative deviation" begin
                m = Model(HiGHS.Optimizer)
                d = semideviation(m, 0.5, 1.0)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.0
            end
            @testset "positive deviation" begin
                m = Model(HiGHS.Optimizer)
                d = semideviation(m, 1.5, 1.0)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.5
            end
        end
        @testset "semideviation with expressions" begin
            @testset "no deviation" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x)
                d = semideviation(m, x, 1.0)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.0
            end
            @testset "negative deviation" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x <= 0.5)
                d = semideviation(m, x, 1.0)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.0
            end
            @testset "positive deviation" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x >= 1.5)
                d = semideviation(m, x, 1.0)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.5
            end
        end
    end
end

function test_dispersion_metrics()
    @testset "dispersion metrics" begin
        @testset "max semideviation" begin
            @testset "no dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 10,
                    :b => 10,
                    :c => 10
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:max_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0
            end
            @testset "big positive dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 0,
                    :b => 20,
                    :c => 80
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:max_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 50.0
            end
            @testset "big negative dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 0,
                    :b => 100,
                    :c => 120
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:max_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 40.0
            end
            @testset "no scenarios" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:max_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.0
            end
        end
        @testset "average semideviation" begin
            @testset "no dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 10,
                    :b => 10,
                    :c => 10
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0
            end
            @testset "big positive dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 0,
                    :b => 20,
                    :c => 80
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 12.5
            end
            @testset "big negative dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 0,
                    :b => 100,
                    :c => 120
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 20.0
            end
            @testset "no scenarios" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(

                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_semideviation))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.0
            end
        end
        @testset "gini difference" begin
            @testset "no dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 10,
                    :b => 10,
                    :c => 10
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_gini_difference))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0
            end
            @testset "big positive dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 0,
                    :b => 20,
                    :c => 80
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_gini_difference))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 15.0
            end
            @testset "big negative dispersion" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(
                    :a => 0,
                    :b => 100,
                    :c => 120
                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_gini_difference))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 22.5
            end
            @testset "no scenarios" begin
                m = Model(HiGHS.Optimizer)
                scenario_costs = Dict(

                )
                scenario_probabilities = Dict(:a => 1/4, :b=> 1/2, :c => 1/4)
                prob = (i) -> scenario_probabilities[i]
                d = dispersion_metric(m, scenario_costs, prob, Val(:avg_gini_difference))
                @objective(m, Min, d)
                set_silent(m)
                optimize!(m)
                @test value(d) == 0.0
            end
        end
    end
end

function test_expected_value_optimization()
    @testset "expected value optimization" begin
        m = Model(HiGHS.Optimizer)
        @variable(m, x[1:2], Bin)
        @constraint(m, x[1] + x[2] == 1)
        scenario_costs = Dict(
            :a => x[1] + 2x[2],
            :b => 100x[1] + 2x[2] 
        )
        scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
        prob = (i) -> scenario_probabilities[i]
        @objective(m, Min, expected_value(scenario_costs, prob))
        set_silent(m)
        optimize!(m)
        @test value(x[1]) == 1.0
        @test value(x[2]) == 0.0
    end
end

function test_cvar_optimization()
    @testset "cvar optimization" begin
        @testset "invalid beta parameter" begin
            m = Model(HiGHS.Optimizer)
            @variable(m, x[1:2], Bin)
            @constraint(m, x[1] + x[2] == 1)
            scenario_costs = Dict(
                :a => x[1] + 2x[2],
                :b => 100x[1] + 2x[2] 
            )
            scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
            prob = (i) -> scenario_probabilities[i]
            @test_throws DomainError cvar(m, 0., scenario_costs, prob)
            @test_throws DomainError cvar(m, 2., scenario_costs, prob)
        end
        @testset "risk-averse cvar" begin
            m = Model(HiGHS.Optimizer)
            @variable(m, x[1:2], Bin)
            @constraint(m, x[1] + x[2] == 1)
            scenario_costs = Dict(
                :a => x[1] + 2x[2],
                :b => 100x[1] + 2x[2] 
            )
            scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
            prob = (i) -> scenario_probabilities[i]
            @objective(m, Min, cvar(m, 0.01, scenario_costs, prob))
            set_silent(m)
            optimize!(m)
            @test isapprox(value(x[1]), 0)
            @test isapprox(value(x[2]), 1)
        end
        @testset "less risk-averse cvar" begin
            m = Model(HiGHS.Optimizer)
            @variable(m, x[1:2], Bin)
            @constraint(m, x[1] + x[2] == 1)
            scenario_costs = Dict(
                :a => x[1] + 2x[2],
                :b => 100x[1] + 2x[2] 
            )
            scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
            prob = (i) -> scenario_probabilities[i]
            @objective(m, Min, cvar(m, 1.0, scenario_costs, prob))
            set_silent(m)
            optimize!(m)
            @test value(x[1]) == 1.0
            @test value(x[2]) == 0.0
        end
    end
end

function test_markowitz_optimization()
    @testset "markowitz optimization" begin
        @testset "markowitz max semideviation" begin
            @testset "invalid parameters" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @test_throws DomainError markowitz_model(m, 0., scenario_costs, prob, Val(:max_semideviation))
                @test_throws DomainError markowitz_model(m, 1., scenario_costs, prob, Val(:max_semideviation))
            end
            @testset "risk-averse" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @objective(m, Min, markowitz_model(m, 0.8, scenario_costs, prob, Val(:max_semideviation)))
                set_silent(m)
                optimize!(m)
                @test isapprox(value(x[1]), 0)
                @test isapprox(value(x[2]), 1)
            end
            @testset "less risk-averse" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @objective(m, Min, markowitz_model(m, 1e-6, scenario_costs, prob, Val(:max_semideviation)))
                set_silent(m)
                optimize!(m)
                @test value(x[1]) == 1.
                @test value(x[2]) == 0.
            end
        end
        @testset "markowitz avg semideviation" begin
            @testset "invalid parameters" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @test_throws DomainError markowitz_model(m, 0., scenario_costs, prob, Val(:avg_semideviation))
                @test_throws DomainError markowitz_model(m, 1., scenario_costs, prob, Val(:avg_semideviation))
            end
            @testset "risk-averse" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @objective(m, Min, markowitz_model(m, 0.8, scenario_costs, prob, Val(:avg_semideviation)))
                set_silent(m)
                optimize!(m)
                @test isapprox(value(x[1]), 0.)
                @test isapprox(value(x[2]), 1.)
            end
            @testset "less risk-averse" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @objective(m, Min, markowitz_model(m, 0.001, scenario_costs, prob, Val(:avg_semideviation)))
                set_silent(m)
                optimize!(m)
                @test value(x[1]) == 1.
                @test value(x[2]) == 0.
            end
        end
        @testset "markowitz gini difference" begin
            @testset "invalid parameters" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @test_throws DomainError markowitz_model(m, 0., scenario_costs, prob, Val(:avg_gini_difference))
                @test_throws DomainError markowitz_model(m, 1., scenario_costs, prob, Val(:avg_gini_difference))
            end
            @testset "risk-averse" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2], Bin)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 2x[2],
                    :b => 100x[1] + 2x[2] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @objective(m, Min, markowitz_model(m, 0.8, scenario_costs, prob, Val(:avg_gini_difference)))
                set_silent(m)
                optimize!(m)
                @test isapprox(value(x[1]), 0.)
                @test isapprox(value(x[2]), 1.)
            end
            @testset "less risk-averse" begin
                m = Model(HiGHS.Optimizer)
                @variable(m, x[1:2] >= 0)
                @constraint(m, x[1] + x[2] == 1)
                scenario_costs = Dict(
                    :a => x[1] + 10x[2],
                    :b => 100x[1] 
                )
                scenario_probabilities = Dict(:a => 0.99, :b => 0.01)
                prob = (i) -> scenario_probabilities[i]
                @objective(m, Min, markowitz_model(m, 0.001, scenario_costs, prob, Val(:avg_gini_difference)))
                set_silent(m)
                optimize!(m)
                @test value(x[1]) == 1.
                @test value(x[2]) == 0.
            end
        end
    end
end

function setup_risk_approach_case(risk_approach="expected_value"; cvar_percentage=0.5, dispersion_measure="max_semideviation", lambda=0.5)
    url_in = "sqlite://"
    file_path_out = "$(@__DIR__)/test_out.sqlite"
    url_out = "sqlite:///$file_path_out"
    objects = [
        ["model", "simple"],
        ["report", "report1"],
        ["stochastic_scenario", "parent"],
        ["stochastic_scenario", "child1"],
        ["stochastic_scenario", "child2"],
        ["stochastic_structure", "DAG"],
        ["temporal_block", "investment"],
        ["temporal_block", "operation2030"],
        ["report", "report1"],
        ["node", "fuel_node"],
        ["node", "electricity_node"],
        ["unit", "safe_plant"],
        ["unit", "risky_plant"]
    ]
    relationships = [
        ["model__default_investment_stochastic_structure", ["simple", "DAG"]],
        ["model__default_investment_temporal_block", ["simple", "investment"]],
        ["model__default_stochastic_structure", ["simple", "DAG"]],
        ["model__default_temporal_block", ["simple", "operation2030"]],
        ["model__report", ["simple", "report1"]],
        ["model__stochastic_structure", ["simple", "DAG"]],
        ["model__temporal_block", ["simple", "investment"]],
        ["model__temporal_block", ["simple", "operation2030"]],
        ["stochastic_structure__stochastic_scenario", ["DAG", "parent"]],
        ["stochastic_structure__stochastic_scenario", ["DAG", "child1"]],
        ["stochastic_structure__stochastic_scenario", ["DAG", "child2"]],
        ["report__output", ["report1", "unit_flow"]],
        ["unit__from_node", ["safe_plant", "fuel_node"]],
        ["unit__from_node", ["risky_plant", "fuel_node"]],
        ["unit__to_node", ["safe_plant", "electricity_node"]],
        ["unit__to_node", ["risky_plant", "electricity_node"]],
        ["unit__node__node", ["safe_plant", "electricity_node", "fuel_node"]],
        ["unit__node__node", ["risky_plant", "electricity_node", "fuel_node"]],
        ["parent_stochastic_scenario__child_stochastic_scenario", ["parent", "child1"]],
        ["parent_stochastic_scenario__child_stochastic_scenario", ["parent", "child2"]],
    ]
    object_parameter_values = [
        ["model", "simple", "risk_approach", risk_approach],
        ["model", "simple", "cvar_percentage", cvar_percentage],
        ["model", "simple", "dispersion_measure", dispersion_measure],
        ["model", "simple", "dispersion_to_mean_cost_weight", lambda],
        ["model", "simple", "model_end", Dict("data" => "2031-01-01T00:00:00.0", "type" => "date_time")],
        ["model", "simple", "model_start", Dict("data" => "2030-01-01T00:00:00.0", "type" => "date_time")],
        ["node", "fuel", "balance_type", "balance_type_none"],
        ["node", "electricity_node", "demand", 100.0],
        ["node", "fuel_node", "balance_type","balance_type_none"],
        ["temporal_block", "investment", "resolution", Dict("data"=> "6M", "type"=> "duration")],
        ["temporal_block", "operation2030", "block_end", Dict( "data" => "2031-01-01T00:00:00.0", "type"=> "date_time")],
        ["temporal_block", "operation2030", "block_start", Dict("data"=> "2030-01-01T00:00:00.0", "type"=> "date_time")],
        ["temporal_block", "operation2030", "resolution", Dict("data"=> "6M", "type"=> "duration")],
        ["unit", "safe_plant", "candidate_units", Dict("data"=> Dict(
                    "2030-01-01T00:00:00.0"=> 1.0,
                    "2030-07-01T00:00:00.0"=> 0.0
                ),
                "type"=> "time_series"
            )
        ],
        ["unit", "safe_plant", "initial_units_invested_available", 0],
        ["unit", "safe_plant", "number_of_units", 0],
        ["unit", "safe_plant", "unit_investment_cost", 1e9],
        ["unit", "safe_plant", "unit_investment_variable_type", "unit_investment_variable_type_integer"],
        ["unit","safe_plant","units_on_cost",0],
        ["unit", "risky_plant", "candidate_units", Dict("data"=> Dict(
                    "2030-01-01T00:00:00.0"=> 1.0,
                    "2030-07-01T00:00:00.0"=> 0.0
                ),
                "type"=> "time_series"
            )
        ],
        ["unit", "risky_plant", "initial_units_invested_available", 0],
        ["unit", "risky_plant", "number_of_units", 0],
        ["unit", "risky_plant", "unit_investment_cost", 1e9],
        ["unit", "risky_plant", "unit_investment_variable_type", "unit_investment_variable_type_integer"],
        ["unit","risky_plant","units_on_cost",0],
    ]
    relationship_parameter_values = [
        [
        "unit__from_node", ["safe_plant", "fuel_node"], "vom_cost", Dict(
            "data" => [["parent", 52], ["child1", 52], ["child2", 52]],
            "type" => "map",
            "index_type" => "str"
        ),],
        [
            "unit__from_node", ["risky_plant", "fuel_node"], "vom_cost",
            Dict(
                "data" => [["parent", 50], ["child1", 30], ["child2", 70]],
                "type" => "map",
                "index_type" => "str"
            ),
            ],
        ["unit__node__node", ["safe_plant", "electricity_node", "fuel_node"], "fix_ratio_out_in_unit_flow", 0.7],
        ["unit__to_node", ["safe_plant", "electricity_node"], "unit_capacity", 100],
        ["unit__node__node", ["risky_plant", "electricity_node", "fuel_node"], "fix_ratio_out_in_unit_flow", 0.7],
        ["unit__to_node", ["risky_plant", "electricity_node"], "unit_capacity", 100],
        ["stochastic_structure__stochastic_scenario", ["DAG", "child1"], "weight_relative_to_parents", 0.5],
        ["stochastic_structure__stochastic_scenario", ["DAG", "child2"], "weight_relative_to_parents", 0.5],
        ["stochastic_structure__stochastic_scenario", ["DAG", "parent"], "stochastic_scenario_end", Dict("type" => "duration", "data" => "6M")],
    
    
    ]
    test_data = Dict(
        :objects => objects,
        :relationships => relationships,
        :object_parameter_values =>  object_parameter_values,
        :relationship_parameter_values => relationship_parameter_values
    )
    _load_test_data(url_in, test_data)
    url_in, url_out, file_path_out
end

function should_be_invested(m::Model, to_invest::Symbol)
    @fetch units_invested = m.ext[:spineopt].variables
    for (k,v) in units_invested
        if (
            k.unit == Object(to_invest, :unit) &&
            k.stochastic_scenario == Object(:parent, :stochastic_scenario) &&
            k.t.start[] == DateTime(2030,1,1) &&
            k.t.end_[] == DateTime(2030,7,1)
        )
            @test isapprox(value(v), 1)
        else
            @test isapprox(value(v), 0) 
        end
    end
end
function test_expected_value_call()
    @testset "expected value call" begin
        url_in, url_out, file_path_out = setup_risk_approach_case("expected_value")
        m = run_spineopt(url_in, url_out; log_level=1)
        should_be_invested(m, :risky_plant)
    end
end

function test_cvar_call()
    @testset "cvar call" begin
        @testset "cvar risk averse" begin
            url_in, url_out, file_path_out = setup_risk_approach_case("cvar"; cvar_percentage=0.01)
            m = run_spineopt(url_in, url_out; log_level=1)
            should_be_invested(m, :safe_plant)
        end
        @testset "cvar less risk averse" begin
            url_in, url_out, file_path_out = setup_risk_approach_case("cvar"; cvar_percentage=1.)
            m = run_spineopt(url_in, url_out; log_level=1)
            should_be_invested(m, :risky_plant)
        end

    end
end

function test_markowitz_dispersion_call(dispersion_measure)
    @testset "$dispersion_measure" begin
        @testset "more risk averse" begin
            url_in, url_out, file_path_out = setup_risk_approach_case("markowitz"; dispersion_measure=dispersion_measure, lambda=0.99)
            m = run_spineopt(url_in, url_out; log_level=1)
            should_be_invested(m, :safe_plant)
        end
        @testset "less risk averse" begin
            url_in, url_out, file_path_out = setup_risk_approach_case("markowitz"; dispersion_measure=dispersion_measure, lambda=0.01)
            m = run_spineopt(url_in, url_out; log_level=1)
            should_be_invested(m, :risky_plant)
        end
    end
end

function test_markowitz_call()
    @testset "markowitz call" begin
        test_markowitz_dispersion_call("max_semideviation")
        test_markowitz_dispersion_call("avg_semideviation")
        test_markowitz_dispersion_call("avg_gini_difference")
    end
end

@testset "costs under risk" begin
    test_expected_value()
    test_positive_part()
    test_semideviation()
    test_dispersion_metrics()
    test_expected_value_optimization()
    test_cvar_optimization()
    test_markowitz_optimization()
    test_expected_value_call()
    test_cvar_call()
    test_markowitz_call()
end
