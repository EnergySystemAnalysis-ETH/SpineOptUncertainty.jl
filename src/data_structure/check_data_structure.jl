#############################################################################
# Copyright (C) 2017 - 2020  Spine Project
#
# This file is part of Spine Model.
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

# NOTE: I see some small problem here, related to doing double work.
# For example, checking that the stochastic DAGs have no loops requires to generate those DAGs, 
# but we can't generate them just for checking and then throw them away, can we?
# So I propose we do that type of checks when we actually generate the corresponding structure.
# And here, we just perform simpler checks that can be done directly on the contents of the db, 
# and don't require to build any additional structures.

"""
    _check(cond, err_msg)

Check the conditional `cond` and throws an error with a message `err_msg` if `cond` is `false`.
"""
_check(cond, err_msg) = cond || error(err_msg)

"""
    check_data_structure(log_level::Int64)

Check if the data structure provided from the db results in a valid model.
"""
function check_data_structure(log_level::Int64)
    check_model_object()
    check_temporal_block_object()
    check_node__stochastic_structure()
    check_unit__stochastic_structure()
    check_minimum_operating_point_unit_capacity()
    check_islands(log_level)
end

"""
    check_model_object()

Check if at least one `model` object is defined.
"""
function check_model_object()
    _check(
        !isempty(model()),
        "`model` object not found - you need a `model` object to run Spine Opt"
    )
end

"""
    check_temporal_block_object()

Check if at least `temporal_block` is defined.
"""
function check_temporal_block_object()
    _check(
        !isempty(temporal_block()),
        "`temporal_block` object not found - you need at least one `temporal_block` to run Spine Opt"
    )
end

"""
    check_node__stochastic_structure()

Ensure there's exactly one `node__stochastic_structure` definition per `node` in the data.
"""
function check_node__stochastic_structure()
    error_nodes = [n for n in node() if length(node__stochastic_structure(node=n)) != 1]
    _check(
        isempty(error_nodes),
        "invalid `node__stochastic_structure` definition for `node`(s): $(join(error_nodes, ", ", " and ")) "
        * "- each `node` must be related to one and only one `stochastic_structure`"
    )
end

"""
    check_unit__stochastic_structure()

Ensure there's exactly one `units_on__stochastic_structure` definition per `unit` in the data.
"""
function check_unit__stochastic_structure()
    error_units = [u for u in unit() if length(units_on__stochastic_structure(unit=u)) != 1]
    _check(
        isempty(error_units),
        "invalid `units_on__stochastic_structure` definition for `unit`(s): $(join(error_units, ", ", " and ")) "
        * "- each `unit` must be related to one and only one `stochastic_structure`"
    )
end

"""
    check_minimum_operating_point_unit_capacity()

Check if every defined `minimum_operating_point` parameter has a corresponding `unit_capacity` parameter defined.
"""
function check_minimum_operating_point_unit_capacity()
    error_indices = [
        (u, n, d) 
        for (u, n, d) in indices(minimum_operating_point) 
        if unit_capacity(unit=u, node=n, direction=d) === nothing
    ]
    _check(
        isempty(error_indices),
        "missing `unit_capacity` value for indices: $(join(error_indices, ", ", " and ")) "
        * "- `unit_capacity` must be specified where `minimum_operating_point` is"
    )
end

"""
    check_islands()

Check network for islands and warn the user if problems.
"""
function check_islands(log_level)

    level0 = log_level >= 0
    level1 = log_level >= 1
    level2 = log_level >= 2
    level3 = log_level >= 3

    for c in commodity()
        if commodity_physics(commodity=c) in (:commodity_physics_ptdf, :commodity_physics_lodf)
            @logtime level3 "Checking network of commodity $(c) for islands" n_islands, island_node = islands(c)
            @log     level3 "The network consists of $(n_islands) islands"
            if n_islands > 1
                @warn "the network of commodity $(c) consists of multiple islands, this may end badly..."
                # add diagnostic option to print island_node which will tell the user which nodes are in which islands
            end
        end
    end
end

"""
    islands()

Determine the number of islands in a commodity network - used for diagnostic purposes.
"""
function islands(c)
    visited_d = Dict{Object,Bool}()
    island_node = Dict{Int64,Array}()
    island_count = 0

    for n in node__commodity(commodity=c)
        visited_d[n] = false
    end

    for n in node__commodity(commodity=c)
        if !visited_d[n]
            island_count = island_count + 1
            island_node[island_count] = Object[]
            visit(n, island_count, visited_d, island_node)
        end
    end
    island_count, island_node
end

"""
    visit()

Recursively visit nodes in the network to determine number of islands.
"""
function visit(n, island_count, visited_d, island_node)
    visited_d[n] = true
    push!(island_node[island_count], n)
    for (conn, n2) in connection__node__node(node1=n)
        if !visited_d[n2]
            visit(n2, island_count, visited_d, island_node)
        end
    end
end

"""
    check_x()

Check for low reactance values.
"""
function check_x()
    @info "Checking reactances"
    for conn in connection()
        if conn_reactance(connection=conn) < 0.0001
            @info "low reactance may cause problems for line " conn
        end
    end
end
