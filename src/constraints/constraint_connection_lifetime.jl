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
    add_constraint_connection_lifetime!(m::Model)

Constrain connections_invested_available by the investment lifetime of a connection.
"""
function add_constraint_connection_lifetime!(m::Model)
    _add_constraint!(
        m, :connection_lifetime, constraint_connection_lifetime_indices, _build_constraint_connection_lifetime
    )
end

function _build_constraint_connection_lifetime(m::Model, conn, s_path, t)
    @fetch connections_invested_available, connections_invested = m.ext[:spineopt].variables
    @build_constraint(
        sum(
            connections_invested_available[conn, s, t]
            for (conn, s, t) in connections_invested_available_indices(
                m; connection=conn, stochastic_scenario=s_path, t=t
            );
            init=0,
        )
        >=
        sum(
            connections_invested[conn, s_past, t_past] * weight
            for (conn, s_past, t_past, weight) in _past_connections_invested_available_indices(m, conn, s_path, t)
        )
    )
end

function constraint_connection_lifetime_indices(m::Model)
    (
        (connection=conn, stochastic_path=path, t=t)
        for (conn, t) in connection_investment_time_indices(m; connection=indices(connection_investment_tech_lifetime))
        for path in active_stochastic_paths(m, _past_connections_invested_available_indices(m, conn, anything, t))
    )
end

function _past_connections_invested_available_indices(m, conn, s_path, t)
    _past_indices(m, connections_invested_available_indices, connection_investment_tech_lifetime, s_path, t; connection=conn)
end

"""
    constraint_connection_lifetime_indices_filtered(m::Model; filtering_options...)

Form the stochastic indexing Array for the `:connections_invested_lifetime()` constraint.

Uses stochastic path indexing due to the potentially different stochastic structures between present and past time.
Keyword arguments can be used to filther the resulting Array.
"""
function constraint_connection_lifetime_indices_filtered(
    m::Model;
    connection=anything,
    stochastic_path=anything,
    t=anything,
)
    f(ind) = _index_in(ind; connection=connection, stochastic_path=stochastic_path, t=t)
    filter(f, constraint_connection_lifetime_indices(m))
end
