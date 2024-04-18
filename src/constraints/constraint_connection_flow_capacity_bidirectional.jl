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

@doc raw"""
When a [connection](@ref) linking to a [node](@ref) is bidirectionally bounded (the [connection\_capacity](@ref)s 
of both directions are defined), a compact linear constraint is generated to ensure that the simultaneous flows 
in both directions do not exceed their own capacity nor does their sum exceed the capacity in each direction.

```math
\begin{aligned}
& p^{connection\_capacity}_{(conn,ng,d\_reverse,s,t)} \cdot 
  p^{connection\_conv\_cap\_to\_flow}_{(conn,ng,d\_reverse,s,t)} \cdot
  \sum_{n \in ng} v^{connection\_flow}_{(conn,n,d,s,t)} \\
& + \\ 
& p^{connection\_capacity}_{(conn,ng,d,s,t)} \cdot 
  p^{connection\_conv\_cap\_to\_flow}_{(conn,ng,d,s,t)} \cdot 
  \sum_{n \in ng} v^{connection\_flow}_{(conn,n,d\_reverse,s,t)} \\
& \leq p^{connection\_availability\_factor}_{(conn,s,t)} \\
& \cdot p^{connection\_capacity}_{(conn,ng,d,s,t)} \cdot 
  p^{connection\_conv\_cap\_to\_flow}_{(conn,ng,d,s,t)} \\
& \cdot p^{connection\_capacity}_{(conn,ng,d\_reverse,s,t)} \cdot 
  p^{connection\_conv\_cap\_to\_flow}_{(conn,ng,d\_reverse,s,t)} \\
& \cdot \begin{cases}       
   v^{connections\_invested\_available}_{(conn,s,t)} 
   & \text{if } p^{candidate\_connections}_{(conn,s,t)} \geq 1 \\
   1 & \text{otherwise} \\
\end{cases} \\
& \forall (conn, ng, d, d\_reverse): \\
& \quad \text{(1) } (conn, ng, d) \in indices(p^{connection\_capacity}) \\
& \qquad \land (conn, ng, d\_reverse) \in indices(p^{connection\_capacity}) \\ 
& \forall (s,t) \\
& \text{where:} \\
& \text{(i) } \text{(1)} \Rightarrow \exist d \land \exist d\_reverse \\
& \qquad \ \ \ \Rightarrow \exist p^{connection\_capacity}_{(conn,ng,d,s,t)} \land \exist p^{connection\_capacity}_{(conn,ng,d\_reverse,s,t)} \\
& \text{(ii) } \exist x \Leftrightarrow x \neq nothing \\
\end{aligned}
```

See also
[connection\_capacity](@ref),
[connection\_availability\_factor](@ref),
[connection\_conv\_cap\_to\_flow](@ref),
[candidate\_connections](@ref)

!!! note
    The conversion factor [connection\_conv\_cap\_to\_flow](@ref) has a default value of `1`,
    but can be adjusted in case the unit of measurement for the capacity is different to the connection flows
    unit of measurement.

"""
# FIXME: When the connection capacity of one direction is 0, 
# the constraint duplicates the one generated by `constraint_connection_flow_capacity()` 
function add_constraint_connection_flow_capacity_bidirectional!(m::Model)
    @fetch connection_flow, connections_invested_available = m.ext[:spineopt].variables
    t0 = _analysis_time(m)
    m.ext[:spineopt].constraints[:connection_flow_capacity_bidirectional] = Dict(
        (connection=conn, node=ng, stochastic_path=s_path, t=t) => @constraint(
            m,
            + sum(
                connection_flow[conn, n, d, s, t] * duration(t)
                for (conn, n, d, s, t) in connection_flow_indices(
                    m; connection=conn, direction=d, node=ng, stochastic_scenario=s_path, t=t_in_t(m; t_long=t)
                );
                init=0,
            ) 
            * sum(
                + connection_capacity[
                    (connection=conn, node=ng, direction=d_reverse, stochastic_scenario=s, analysis_time=t0, t=t)
                ] 
                * connection_conv_cap_to_flow[
                    (connection=conn, node=ng, direction=d_reverse, stochastic_scenario=s, analysis_time=t0, t=t)
                ]
                for (conn, _n, d_reverse, s, t) in connection_flow_indices(
                    m; connection=conn, direction=d_reverse, node=ng, stochastic_scenario=s_path, t=t_in_t(m; t_long=t)
                );
                init=0,
            )
            + sum(
                connection_flow[conn, n, d_reverse, s, t] * duration(t)
                for (conn, n, d_reverse, s, t) in connection_flow_indices(
                    m; connection=conn, direction=d_reverse, node=ng, stochastic_scenario=s_path, t=t_in_t(m; t_long=t)
                );
                init=0,
            ) 
            * sum(
                + connection_capacity[
                    (connection=conn, node=ng, direction=d, stochastic_scenario=s, analysis_time=t0, t=t)
                ] 
                * connection_conv_cap_to_flow[
                    (connection=conn, node=ng, direction=d, stochastic_scenario=s, analysis_time=t0, t=t),
                ]
                for (conn, _n, d, s, t) in connection_flow_indices(
                    m; connection=conn, direction=d, node=ng, stochastic_scenario=s_path, t=t_in_t(m; t_long=t)
                );
                init=0,                
            )
            <=
            + maximum(
                connection_availability_factor[(connection=conn, stochastic_scenario=s, analysis_time=t0, t=t)]
                for s in s_path
            )
            * sum(
                + connection_capacity[
                    (connection=conn, node=ng, direction=d, stochastic_scenario=s, analysis_time=t0, t=t)
                ] 
                * connection_conv_cap_to_flow[
                    (connection=conn, node=ng, direction=d, stochastic_scenario=s, analysis_time=t0, t=t),
                ]
                for (conn, _n, d, s, t) in connection_flow_indices(
                    m; connection=conn, direction=d, node=ng, stochastic_scenario=s_path, t=t_in_t(m; t_long=t)
                );
                init=0,
            )
            * sum(
                + connection_capacity[
                    (connection=conn, node=ng, direction=d_reverse, stochastic_scenario=s, analysis_time=t0, t=t)
                ] 
                * connection_conv_cap_to_flow[
                    (connection=conn, node=ng, direction=d_reverse, stochastic_scenario=s, analysis_time=t0, t=t)
                ]
                for (conn, _n, d_reverse, s, t) in connection_flow_indices(
                    m; connection=conn, direction=d_reverse, node=ng, stochastic_scenario=s_path, t=t_in_t(m; t_long=t)
                );
                init=0,
            )
            * (
                !isnothing(candidate_connections(connection=conn)) ? sum(
                    connections_invested_available[conn, s, t1]
                    for (conn, s, t1) in connections_invested_available_indices(
                        m; connection=conn, stochastic_scenario=s_path, t=t_in_t(m; t_short=t)
                    );
                    init=0,
                ) : 1
            )
            * duration(t)
        )
        for (conn, ng, d, d_reverse, s_path, t) in constraint_connection_flow_capacity_bidirectional_indices(m)
    )
end

function constraint_connection_flow_capacity_bidirectional_indices(m::Model)    
    (
        (connection=conn, node=ng, direction=d, reverse_direction=_d_reverse(d), stochastic_path=path, t=t)
        for (conn, ng, d) in indices(connection_capacity)
        if connection_capacity(connection=conn, node=ng, direction=_d_reverse(d), _strict=false) !== nothing
        for (t, path) in t_lowest_resolution_path(
            m,
            connection_flow_indices(m; connection=conn, node=ng),
            connections_invested_available_indices(m; connection=conn),
        )
    )
end