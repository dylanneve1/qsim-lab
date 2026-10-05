# The [[288,16,16]] code: G = SmallGroup(144,167) = C6 x K, K = SmallGroup(24,8) = C3 : D8.
# Chooses z (order 6, central factor), a (order 3), b (order 4), c (order 2) with
# K = <a, b, c>, prints their relations, and writes every element of A and B
# (indices in the numbering of export_groups.g) in the normal form z^i a^j b^k c^l.
LoadPackage("smallgrp");
G := SmallGroup(144,167);
els := ShallowCopy(AsSSortedList(G));
e := Position(els, One(G)); els[e] := els[1]; els[1] := One(G);
dfs := DirectFactorsOfGroup(G);
C2f := First(dfs, f -> Size(f) = 2); C3f := First(dfs, f -> Size(f) = 3); K := First(dfs, f -> Size(f) = 24);
z := MinimalGeneratingSet(C2f)[1] * MinimalGeneratingSet(C3f)[1];
a := MinimalGeneratingSet(SylowSubgroup(K, 3))[1];
P := SylowSubgroup(K, 2);
b := First(AsList(P), x -> Order(x) = 4);
c := First(AsList(P), x -> Order(x) = 2 and not x in Group(b) and x^-1*b*x = b^-1);
Print("K = ", StructureDescription(K), " Sylow-2 = ", StructureDescription(P), "\n");
Print("orders z,a,b,c: ", List([z,a,b,c], Order), "\n");
Print("b^-1 a b = a^", First([1,2], i -> b^-1*a*b = a^i), ", c a c = a^", First([1,2], i -> c*a*c = a^i),
      ", c b c = b^", First([1,2,3], i -> c*b*c = b^i), "\n");
nf := function(x)
  local i, j, k, l;
  for i in [0..5] do for j in [0..2] do for k in [0..3] do for l in [0..1] do
    if z^i*a^j*b^k*c^l = x then return [i, j, k, l]; fi;
  od; od; od; od;
  return fail;
end;
Print("normal forms unique: ", Size(Set(List(els, nf))) = 144, "\n");
Print("generator indices z,a,b,c: ", List([z,a,b,c], x -> Position(els, x) - 1), "\n");
Print("A = ", List([0,9,83], i -> nf(els[i+1])), "  (z^i a^j b^k c^l as [i,j,k,l])\n");
Print("B = ", List([0,51,90], i -> nf(els[i+1])), "\n");
QUIT;
