
# specifiy primitive functions
neg(a) = a*(-1); 
squ(a) = a^2; 
div(a,b) = /(a,b); 
inv(a) = div(1,a); 
mul(a,b) = a*b; 
add(a,b) = a+b; 
sub(a,b) = a-b;

# conditional primitives tenary
ifg0(a,b,c) = ifelse(a>0,b,c);
ifge0(a,b,c) = ifelse(a>=0,b,c);
if0(a,b,c) = ifelse(a==0,b,c);
ifle0(a,b,c) = ifelse(a<=0,b,c);
ifl0(a,b,c) = ifelse(a<0,b,c);

# arity = Dict(:add => 2, :sub => 2, :mul => 2, :div => 2, :min => 2, :max => 2, :inv => 1, :neg => 1, :squ => 1)

# #define primitives HERE
# PRIMITIVES = [add,sub,mul,div,min,max,neg,inv,squ]
# primitives1 = [p for p in PRIMITIVES if arity[Symbol(p)] == 1]
# primitives2 = [p for p in PRIMITIVES if arity[Symbol(p)] == 2]

# primitivesSymbol = Dict(f => string(f) for f in PRIMITIVES)

# FUN_STRING_DICT = Dict(add=>"+", sub=>"-", mul=>"*", div=>"/", min=>"min", max=>"max", neg=>"neg", inv=>"inv", squ=>"squ")
# STRING_FUN_DICT = Dict(v=>k for (k,v) in FUN_STRING_DICT)
