#############################################################################
# Copyright (C) 2017 - 2018  Spine Project
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
"""
    node_state_indices(filtering_options...)

A set of tuples for indexing the `node_state` variable. Any filtering options can be specified
for `node`, `commodity`, and `t`.
"""
function node_state_indices(;node=anything, commodity=anything, t=anything)
    inds = NamedTuple{(:node, :commodity, :t),Tuple{Object,Object,TimeSlice}}[
        (node=n, commodity=c, t=t)
        for (n, c, tb) in node_state_indices_rc(
            node=node, commodity=commodity, _compact=false
        )
        for t in time_slice(temporal_block=tb)
    ]
    unique!(inds)
end

fix_node_state_(x) = fix_node_state(node=x.node, t=x.t, _strict=false)
node_state_lb(x) = node_state_min(node=x.node)
node_state_ub(x) = node_state_max(node=x.node)

create_variable_node_state!(m::Model) = create_variable!(
    m,
    :node_state,
    node_state_indices;
    lb=node_state_lb,
    ub=node_state_ub
)
fix_variable_node_state!(m::Model) = fix_variable!(m, :stor_state, stor_state_indices, fix_stor_state_)

# TODO: Method for node state? Control through `fix_node_state?`