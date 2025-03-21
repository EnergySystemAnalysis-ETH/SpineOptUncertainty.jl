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
    dispersion_metric

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
        end
        @testset "gini difference" begin

        end
    end
end

@testset "costs under risk" begin
    test_expected_value()
    test_positive_part()
    test_semideviation()
    test_dispersion_metrics()
end
