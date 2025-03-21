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
    positive_part_of_lp_term

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

@testset "costs under risk" begin
    test_expected_value()
end
