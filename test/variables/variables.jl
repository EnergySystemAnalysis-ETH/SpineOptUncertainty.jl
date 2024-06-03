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

function _test_variable_unit_setup()
    url_in = "sqlite://"
    test_data = Dict(
        :objects => [
            ["model", "instance"],
            ["temporal_block", "hourly"],
            ["temporal_block", "investments_hourly"],
            ["temporal_block", "two_hourly"],
            ["stochastic_structure", "deterministic"],
            ["stochastic_structure", "investments_deterministic"],
            ["stochastic_structure", "stochastic"],
            ["unit", "unit_ab"],
            ["node", "node_a"],
            ["node", "node_b"],
            ["node", "node_c"],
            ["node", "node_group_bc"],
            ["stochastic_scenario", "parent"],
            ["stochastic_scenario", "child"],
        ],
        :relationships => [
            ["model__temporal_block", ["instance", "hourly"]],
            ["model__temporal_block", ["instance", "investments_hourly"]],
            ["model__temporal_block", ["instance", "two_hourly"]],
            ["model__stochastic_structure", ["instance", "deterministic"]],
            ["model__stochastic_structure", ["instance", "investments_deterministic"]],
            ["model__stochastic_structure", ["instance", "stochastic"]],
            ["units_on__temporal_block", ["unit_ab", "hourly"]],
            ["units_on__stochastic_structure", ["unit_ab", "stochastic"]],
            ["unit__from_node", ["unit_ab", "node_a"]],
            ["unit__to_node", ["unit_ab", "node_b"]],
            ["unit__to_node", ["unit_ab", "node_c"]],
            ["node__temporal_block", ["node_a", "hourly"]],
            ["node__temporal_block", ["node_b", "two_hourly"]],
            ["node__temporal_block", ["node_c", "hourly"]],
            ["node__stochastic_structure", ["node_a", "stochastic"]],
            ["node__stochastic_structure", ["node_b", "deterministic"]],
            ["node__stochastic_structure", ["node_c", "stochastic"]],
            ["stochastic_structure__stochastic_scenario", ["deterministic", "parent"]],
            ["stochastic_structure__stochastic_scenario", ["investments_deterministic", "parent"]],
            ["stochastic_structure__stochastic_scenario", ["stochastic", "parent"]],
            ["stochastic_structure__stochastic_scenario", ["stochastic", "child"]],
            ["parent_stochastic_scenario__child_stochastic_scenario", ["parent", "child"]],
        ],
        :object_groups => [["node", "node_group_bc", "node_b"], ["node", "node_group_bc", "node_c"]],
        :object_parameter_values => [
            ["model", "instance", "model_start", Dict("type" => "date_time", "data" => "2000-01-01T00:00:00")],
            ["model", "instance", "model_end", Dict("type" => "date_time", "data" => "2000-01-01T02:00:00")],
            ["model", "instance", "duration_unit", "hour"],
            ["model", "instance", "model_type", "spineopt_standard"],
            ["model", "instance", "max_gap", "0.05"],
            ["model", "instance", "max_iterations", "2"],
            ["temporal_block", "hourly", "resolution", Dict("type" => "duration", "data" => "1h")],
            ["temporal_block", "investments_hourly", "resolution", Dict("type" => "duration", "data" => "1h")],
            ["temporal_block", "two_hourly", "resolution", Dict("type" => "duration", "data" => "2h")],
            ["model", "instance", "db_mip_solver", "HiGHS.jl"],
            ["model", "instance", "db_lp_solver", "HiGHS.jl"],
            ["unit", "unit_ab", "units_on_cost", 1],  # Just to have units_on variables
        ],
        :relationship_parameter_values => [
            [
                "stochastic_structure__stochastic_scenario",
                ["stochastic", "parent"],
                "stochastic_scenario_end",
                Dict("type" => "duration", "data" => "1h"),
            ]
        ],
    )
    _load_test_data(url_in, test_data)
    url_in
end

function _test_variable_connection_setup()
    url_in = "sqlite://"
    test_data = Dict(
        :objects => [
            ["model", "instance"],
            ["temporal_block", "hourly"],
            ["temporal_block", "investments_hourly"],
            ["temporal_block", "two_hourly"],
            ["stochastic_structure", "deterministic"],
            ["stochastic_structure", "investments_deterministic"],
            ["stochastic_structure", "stochastic"],
            ["connection", "connection_ab"],
            ["connection", "connection_bc"],
            ["connection", "connection_ca"],
            ["node", "node_a"],
            ["node", "node_b"],
            ["node", "node_c"],
            ["stochastic_scenario", "parent"],
            ["stochastic_scenario", "child"],
        ],
        :relationships => [
            ["model__temporal_block", ["instance", "hourly"]],
            ["model__temporal_block", ["instance", "two_hourly"]],
            ["model__temporal_block", ["instance", "investments_hourly"]],
            ["model__stochastic_structure", ["instance", "deterministic"]],
            ["model__stochastic_structure", ["instance", "stochastic"]],
            ["model__stochastic_structure", ["instance", "investments_deterministic"]],
            ["connection__from_node", ["connection_ab", "node_a"]],
            ["connection__to_node", ["connection_ab", "node_b"]],
            ["connection__from_node", ["connection_bc", "node_b"]],
            ["connection__to_node", ["connection_bc", "node_c"]],
            ["connection__from_node", ["connection_ca", "node_c"]],
            ["connection__to_node", ["connection_ca", "node_a"]],
            ["node__temporal_block", ["node_a", "hourly"]],
            ["node__temporal_block", ["node_b", "two_hourly"]],
            ["node__temporal_block", ["node_c", "hourly"]],
            ["node__stochastic_structure", ["node_a", "stochastic"]],
            ["node__stochastic_structure", ["node_b", "deterministic"]],
            ["node__stochastic_structure", ["node_c", "stochastic"]],
            ["stochastic_structure__stochastic_scenario", ["deterministic", "parent"]],
            ["stochastic_structure__stochastic_scenario", ["stochastic", "parent"]],
            ["stochastic_structure__stochastic_scenario", ["stochastic", "child"]],
            ["stochastic_structure__stochastic_scenario", ["investments_deterministic", "parent"]],
            ["parent_stochastic_scenario__child_stochastic_scenario", ["parent", "child"]],
        ],
        :object_parameter_values => [
            ["model", "instance", "model_start", Dict("type" => "date_time", "data" => "2000-01-01T00:00:00")],
            ["model", "instance", "model_end", Dict("type" => "date_time", "data" => "2000-01-01T02:00:00")],
            ["model", "instance", "duration_unit", "hour"],
            ["model", "instance", "model_type", "spineopt_standard"],
            ["model", "instance", "max_gap", "0.05"],
            ["model", "instance", "max_iterations", "2"],
            ["temporal_block", "hourly", "resolution", Dict("type" => "duration", "data" => "1h")],
            ["temporal_block", "two_hourly", "resolution", Dict("type" => "duration", "data" => "2h")],
            ["model", "instance", "db_mip_solver", "HiGHS.jl"],
            ["model", "instance", "db_lp_solver", "HiGHS.jl"],
        ],
        :relationship_parameter_values => [
            [
                "stochastic_structure__stochastic_scenario",
                ["stochastic", "parent"],
                "stochastic_scenario_end",
                Dict("type" => "duration", "data" => "1h")
            ]
        ],
    )
    _load_test_data(url_in, test_data)
    url_in
end

function test_initial_units_on()
    @testset "initial_units_on" begin
        url_in = _test_variable_unit_setup()
        init_units_on = 123
        object_parameter_values = [
            ["unit", "unit_ab", "initial_units_on", init_units_on],
            ["model", "instance", "roll_forward", unparse_db_value(Hour(1))],
        ]
        SpineInterface.import_data(url_in; object_parameter_values=object_parameter_values)
        m = run_spineopt(url_in; log_level=0, optimize=false)
        var_units_on = m.ext[:spineopt].variables[:units_on]
        for key in keys(var_units_on)
            is_history_t = start(key.t) < model_start(model=m.ext[:spineopt].instance)
            @test is_fixed(var_units_on[key]) == is_history_t
            if is_history_t
                @test fix_value(var_units_on[key]) == init_units_on
            end
        end
    end
end

function test_unit_online_variable_type_none()
    @testset "unit_online_variable_type_none" begin
        url_in = _test_variable_unit_setup()
        unit_availability_factor = 0.5
        object_parameter_values = [
            ["unit", "unit_ab", "unit_availability_factor", unit_availability_factor],
            ["unit", "unit_ab", "online_variable_type", "unit_online_variable_type_none"],
            ["model", "instance", "roll_forward", unparse_db_value(Hour(1))],
        ]
        SpineInterface.import_data(url_in; object_parameter_values=object_parameter_values)
        m = run_spineopt(url_in; log_level=0, optimize=true)
        var_units_on = m.ext[:spineopt].variables[:units_on]
        constraint_u_avail = m.ext[:spineopt].constraints[:units_available]
        scenarios = (stochastic_scenario(:parent), stochastic_scenario(:child))
        time_slices = time_slice(m; temporal_block=temporal_block(:hourly))
        @testset for (s, t) in zip(scenarios, time_slices)
            key = (unit(:unit_ab), s, t)
            @test_throws KeyError var_units_on[key...]
            @test_throws KeyError constraint_u_avail[key...]
        end
    end
end

function test_unit_history_parameters()
    @testset "unit_history_parameters" begin
        min_up_minutes = 120
        min_down_minutes = 180
        scheduled_outage_duration_minutes = 60
        lifetime_minutes = 240
        candidate_units = 3
        
        url_in = _test_variable_unit_setup()
        model_end = Dict("type" => "date_time", "data" => "2000-01-01T05:00:00")
        min_up_time = Dict("type" => "duration", "data" => string(min_up_minutes, "m"))
        min_down_time = Dict("type" => "duration", "data" => string(min_down_minutes, "m"))
        scheduled_outage_duration = Dict("type" => "duration", "data" => string(scheduled_outage_duration_minutes, "m"))
        unit_investment_lifetime = Dict("type" => "duration", "data" => string(lifetime_minutes, "m"))
        object_parameter_values = [
            ["unit", "unit_ab", "min_up_time", min_up_time],
            ["unit", "unit_ab", "min_down_time", min_down_time],
            ["unit", "unit_ab", "candidate_units", candidate_units],
            ["unit", "unit_ab", "scheduled_outage_duration", scheduled_outage_duration],
            ["unit", "unit_ab", "outage_variable_type", "unit_online_variable_type_integer"],
            ["unit", "unit_ab", "unit_investment_lifetime", unit_investment_lifetime],
            ["model", "instance", "model_end", model_end],
        ]
        relationships = [
            ["unit__investment_temporal_block", ["unit_ab", "hourly"]],
            ["unit__investment_stochastic_structure", ["unit_ab", "stochastic"]],
        ]
        SpineInterface.import_data(
            url_in; relationships=relationships, object_parameter_values=object_parameter_values
        )
        m = run_spineopt(url_in; log_level=0, optimize=false)

        var_units_on = m.ext[:spineopt].variables[:units_on]
        var_units_started_up = m.ext[:spineopt].variables[:units_started_up]
        var_units_shut_down = m.ext[:spineopt].variables[:units_shut_down]
        var_units_out_of_service = m.ext[:spineopt].variables[:units_out_of_service]
        var_units_taken_out_of_service = m.ext[:spineopt].variables[:units_taken_out_of_service]
        var_units_invested_available = m.ext[:spineopt].variables[:units_invested_available]
        var_units_invested = m.ext[:spineopt].variables[:units_invested]
        
        @test length(var_units_on) == 8
        @test length(var_units_started_up) == 7
        @test length(var_units_shut_down) == 8
        @test length(var_units_out_of_service) == 6
        @test length(var_units_taken_out_of_service) == 6
        @test length(var_units_invested_available) == 9
        @test length(var_units_invested) == 9

    end
end

function test_connection_history_parameters()
    @testset "constraint_connection_lifetime" begin
        flow_ratio = 0.8
        conn_flow_minutes_delay = 180
        lifetime_minutes = 240
        candidate_connections = 3
        model_end = Dict("type" => "date_time", "data" => "2000-01-01T05:00:00")

        url_in = _test_variable_connection_setup()
        connection_investment_lifetime = Dict("type" => "duration", "data" => string(lifetime_minutes, "m"))
        connection_flow_delay = Dict("type" => "duration", "data" => string(conn_flow_minutes_delay, "m"))
        object_parameter_values = [
            ["connection", "connection_ab", "candidate_connections", candidate_connections],
            ["connection", "connection_ab", "connection_investment_lifetime", connection_investment_lifetime],
            ["model", "instance", "model_end", model_end],
        ]
        relationships = [
            ["connection__investment_temporal_block", ["connection_ab", "hourly"]],
            ["connection__investment_stochastic_structure", ["connection_ab", "stochastic"]],
            ["connection__node__node", ["connection_ab", "node_b", "node_a"]],
        ]
        relationship_parameter_values = [
            ["connection__node__node", ["connection_ab", "node_b", "node_a"], "connection_flow_delay", connection_flow_delay],
            ["connection__node__node", ["connection_ab", "node_b", "node_a"], "fix_ratio_out_in_connection_flow", flow_ratio],
        ]
        SpineInterface.import_data(
            url_in;
            relationships=relationships,
            object_parameter_values=object_parameter_values,
            relationship_parameter_values=relationship_parameter_values,
        )
        m = run_spineopt(url_in; log_level=0, optimize=false)
        var_connection_flow = m.ext[:spineopt].variables[:connection_flow]
        var_connections_invested_available = m.ext[:spineopt].variables[:connections_invested_available]
        var_connections_invested = m.ext[:spineopt].variables[:connections_invested]
        
        @test length(var_connection_flow) == 42
        @test length(var_connections_invested_available) == 9
        @test length(var_connections_invested) == 9
    end
end

@testset "unit-based constraints" begin
    test_initial_units_on()
    test_unit_online_variable_type_none()
    test_unit_history_parameters()
    test_connection_history_parameters()
end
