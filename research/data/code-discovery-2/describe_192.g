# The [[192,12,14]] code: G = SmallGroup(96,17) = C3 : (Q8 : C4).
# Chooses a (order 3, normal), i, j (generating a normal Q8 of the Sylow-2) and
# t (order 4) with P = <i, j, t>, prints the relations and writes A, B (indices
# of export_groups.g) in the normal form a^e i^f j^g t^h with f, g < 2... (any
# element of Q8 as i^f j^g (-1)^s with -1 = i^2).
LoadPackage("smallgrp");
G := SmallGroup(96, 17);
els := ShallowCopy(AsSSortedList(G));
e := Position(els, One(G)); els[e] := els[1]; els[1] := One(G);
a := MinimalGeneratingSet(SylowSubgroup(G, 3))[1];
P := SylowSubgroup(G, 2);
Q := First(NormalSubgroups(P), N -> Size(N) = 8 and IdGroup(N) = [8, 4]);
Print("P = ", StructureDescription(P), ", Q8 normal in P: ", Q <> fail, "\n");
gi := First(AsList(Q), x -> Order(x) = 4);
gj := First(AsList(Q), x -> Order(x) = 4 and not x in Group(gi));
t := First(AsList(P), x -> Order(x) = 4 and Size(Group(Concatenation(GeneratorsOfGroup(Q), [x]))) = 32 and Size(Intersection(Group(x), Q)) = 1);
Print("orders a,i,j,t: ", List([a, gi, gj, t], Order), "\n");
w := function(x) local s;
  for s in Tuples([0..3], 3) do
    if x = gi^s[1] * gj^s[2] * t^s[3] then return s; fi;
  od; return fail; end;
Print("t^-1 i t = i^f j^g t^h with [f,g,h] = ", w(t^-1*gi*t), "; t^-1 j t = ", w(t^-1*gj*t), "\n");
Print("i^-1 a i = a^", First([1,2], k -> gi^-1*a*gi = a^k), ", j^-1 a j = a^", First([1,2], k -> gj^-1*a*gj = a^k),
      ", t^-1 a t = a^", First([1,2], k -> t^-1*a*t = a^k), "\n");
nf := function(x) local e2, s;
  for e2 in [0..2] do for s in Tuples([0..3], 3) do
    if x = a^e2 * gi^s[1] * gj^s[2] * t^s[3] then return [e2, s[1], s[2], s[3]]; fi;
  od; od; return fail; end;
Print("A = ", List([0, 1, 15], k -> nf(els[k + 1])), "  as a^e i^f j^g t^h\n");
Print("B = ", List([0, 20, 85], k -> nf(els[k + 1])), "\n");
QUIT;
