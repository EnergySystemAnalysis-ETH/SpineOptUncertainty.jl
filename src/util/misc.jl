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

# override `get` and `getindex` so we can access our variable dicts with a `Tuple` instead of the actual `NamedTuple`
function Base.get(d::Dict{K,VariableRef}, key::Tuple{Vararg{ObjectLike}}, default) where {J,K<:RelationshipLike{J}}
    Base.get(d, NamedTuple{J}(key), default)
end

function Base.getindex(d::Dict{K,VariableRef}, key::ObjectLike...) where {J,K<:RelationshipLike{J}}
    Base.getindex(d, NamedTuple{J}(key))
end

"""
    @fetch x, y, ... = d

Assign mapping of :x and :y in `d` to `x` and `y` respectively
"""
macro fetch(expr)
    (expr isa Expr && expr.head == :(=)) || error("please use @fetch with the assignment operator (=)")
    keys, dict = expr.args
    values = if keys isa Expr
        Expr(:tuple, [:($dict[$(Expr(:quote, k))]) for k in keys.args]...)
    else
        :($dict[$(Expr(:quote, keys))])
    end
    esc(Expr(:(=), keys, values))
end

expand_unit_group(::Anything) = anything
function expand_unit_group(ugs::X) where X >: Anything
    (u for ug in ugs for u in unit_group__unit(unit1=ug, _default=ug))
end

expand_node_group(::Anything) = anything
function expand_node_group(ngs::X) where X >: Anything
    (n for ng in ngs for n in node_group__node(node1=ng, _default=ng))
end

expand_commodity_group(::Anything) = anything
function expand_commodity_group(cgs::X) where X >: Anything
    (c for cg in cgs for c in commodity_group__commodity(commodity1=cg, _default=cg))
end

macro log(level, msg)
    quote
        if $(esc(level))
            printstyled($(esc(msg)), "\n"; bold=true)
        end
    end
end

macro logtime(level, msg, expr)
    quote
        if $(esc(level))
            @msgtime $(esc(msg)) $(esc(expr))
        else
            $(esc(expr))
        end
    end
end

macro msgtime(msg, expr)
    quote
        printstyled($(esc(msg)); bold=true)
        @time $(esc(expr))
    end
end

sense_constraint(m, lhs, sense::typeof(<=), rhs) = @constraint(m, lhs <= rhs)
sense_constraint(m, lhs, sense::typeof(==), rhs) = @constraint(m, lhs == rhs)
sense_constraint(m, lhs, sense::typeof(>=), rhs) = @constraint(m, lhs >= rhs)
function sense_constraint(m, lhs, sense::Symbol, rhs)
    if sense == :>=
        @constraint(m, lhs >= rhs)
    elseif sense == :<=
        @constraint(m, lhs <= rhs)
    else
        @constraint(m, lhs == rhs)
    end
end

"""
    name_constraints!(m::Model)

Sets constraint names for more useful diagnostic output
"""
function name_constraints!(m::Model)
    for (con_key, cons) in m.ext[:constraints]
        for (inds, con) in cons
            set_name(con, string(con_key,inds))
        end
    end
end

"""
    expr_sum(iter; init::Number)

Sum elements in iter to init in-place, and return the result as a GenericAffExpr.
"""
function expr_sum(iter; init::Number)
    result = AffExpr(init)
    isempty(iter) && return result
    result += first(iter)  # NOTE: This is so result has the right type, e.g., `GenericAffExpr{Call,VariableRef}`
    for item in Iterators.drop(iter, 1)
        add_to_expression!(result, item)
    end
    result
end

function write_ptdfs()
    io = open("ptdfs.csv", "w")
    print(io, "connection,")
    for n in node(has_ptdf=true)
        print(io, string(n), ",")
    end
    print(io, "\n")
    for conn in connection(has_ptdf=true)
        print(io, string(conn), ",")
        for n in node(has_ptdf=true)
            print(io, ptdf(connection=conn, node=n), ",")
        end
        print(io, "\n")
    end
    close(io)
end

function write_lodfs()

    io = open("lodfs.csv", "w")
    print(io, raw"contingency line,from_node,to node,")

    for conn_mon in connection(connection_monitored=true)
        print(io, string(conn_mon), ",")
    end
    print(io, "\n")

    for conn_cont in connection(connection_contingency=true)
        n_from, n_to = connection__from_node(connection=conn_cont)
        print(io, string(conn_cont), ",", string(n_from), ",", string(n_to))
        for conn_mon_ in connection(connection_monitored=true)
            print(io, ",")
            for (conn_cont, conn_mon) in indices(lodf; connection1=conn_cont, connection2=conn_mon_)
                print(io, lodf(connection1=conn_cont, connection2=conn_mon))
            end
        end
        print(io, "\n")
    end
    close(io)
end

"""
    pulldims(input, dims...)

An equivalent dictionary where the given dimensions are pulled from the key to the value.
"""
function pulldims(input::Dict{K,V}, dims::Symbol...) where {K<:NamedTuple,V}
    output = Dict()
    for (key, value) in sort!(OrderedDict(input))
        output_key = (; (k => v for (k, v) in pairs(key) if !(k in dims))...)
        output_value = ((key[dim] for dim in dims)..., value)
        push!(get!(output, output_key, []), output_value)
    end
    output
end

"""
    formulation(d::Dict)

An equivalent dictionary where `JuMP.ConstraintRef` values are replaced by their `String` formulation.
"""
formulation(d::Dict{K,JuMP.ConstraintRef}) where {K} = Dict{K,Any}(k => sprint(show, v) for (k, v) in d)