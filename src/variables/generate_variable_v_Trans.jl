function generate_variable_v_Trans(m::Model)
    @variable(m, v_Trans[c in connection(), i in node(),j in node(), t=1:number_of_timesteps("timer"); [c,i,j] in generate_ConnectionNodePairs()]
)
end
