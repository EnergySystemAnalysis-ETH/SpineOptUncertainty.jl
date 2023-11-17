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
Enforce the operating point flow variable `unit_flow_op` at operating point `i` to use its full capacity
if the subsequent operating point `i+1` is active if parameter [ordered\_unit\_flow\_op](@ref) is set `true`.
The last segment does not need this constraint.

```math
\begin{aligned}
& unit\_flow\_op_{(u, n, d, op, s, t)} \\
& \geq UC_{(u, n, d, s, t)} \cdot UCCTF_{(u, n, d, s, t)} \\
& \cdot \left(OP_{(u, n, op, s, t)} - \begin{cases}       
   OP_{(u, n, op-1, s, t)} & \text{if} op > 1 \\
   0 & \text{otherwise} \\
\end{cases} \right) \\
& \cdot unit\_flow\_op\_active_{(u, n, d, op+1, s, t)} \\
& \forall (u,n,d) \in indices(UC) \cup indices(OP): OUFO_{(u,n,d)} \\
& \forall op \in \{ 1, \ldots, \|OP_{(u,n,d)}\| - 1\} \\
& \forall (s,t)
\end{aligned}
```
where
- ``UC =`` [unit\_capacity](@ref)
- ``UCCTF =`` [unit\_conv\_cap\_to\_flow](@ref)
- ``OP =`` [operating\_points](@ref)
- ``OUFO =`` [ordered\_unit\_flow\_op](@ref)
"""
function add_constraint_unit_flow_op_rank!(m::Model)
    @fetch unit_flow_op, unit_flow_op_active = m.ext[:spineopt].variables
    t0 = _analysis_time(m)
    m.ext[:spineopt].constraints[:unit_flow_op_rank] = Dict(
        (unit=u, node=n, direction=d, i=op, stochastic_scenario=s, t=t) => @constraint(
            m,
            + unit_flow_op[u, n, d, op, s, t]
            >=
            (
                + operating_points[(unit=u, node=n, direction=d, stochastic_scenario=s, analysis_time=t0, i=op)] 
                - (
                    (op > 1) ?
                    operating_points[(unit=u, node=n, direction=d, stochastic_scenario=s, analysis_time=t0, i=op - 1)] :
                    0
                )
            )
            * unit_capacity[(unit=u, node=n, direction=d, stochastic_scenario=s, analysis_time=t0, t=t)]
            * unit_conv_cap_to_flow[(unit=u, node=n, direction=d, stochastic_scenario=s, analysis_time=t0, t=t)]
            * unit_flow_op_active[u, n, d, op + 1, s, t]
        )
        for (u, n, d) in indices(unit_capacity)
        for (u, n, d, op, s, t) in unit_flow_op_active_indices(m; unit=u, node=n, direction=d)
        if op < lastindex(operating_points(unit=u, node=n, direction=d))
        # the partial unit flow at the last operating point does not need this constraint.
    )
end
