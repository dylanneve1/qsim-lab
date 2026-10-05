# Describe a code found by group_codes: the group SmallGroup(N, id) with the
# element numbering of export_groups.g (AsSSortedList with the identity moved
# to index 0), a minimal generating set (with element orders and the
# structure of the group), and A, B as words in those generators
# (GAP's Factorization, a shortest word).
# usage: (echo 'N:=84;; id:=13;; A:=[0,11,47];; B:=[0,6,43];;'; cat describe_code.g) | gap -q
LoadPackage("smallgrp");
G := SmallGroup(N, id);
els := ShallowCopy(AsSSortedList(G));
e := Position(els, One(G)); els[e] := els[1]; els[1] := One(G);
gens := MinimalGeneratingSet(G);
Print("group: SmallGroup(", N, ",", id, ") = ", StructureDescription(G), "\n");
Print("generators x1..x", Length(gens), ": element indices ", List(gens, x -> Position(els, x) - 1),
      ", orders ", List(gens, Order), "\n");
Print("commutation table [xi,xj] as words: ");
Gr := GroupWithGenerators(gens);
hom := EpimorphismFromFreeGroup(Gr : names := List([1..Length(gens)], i -> Concatenation("x", String(i))));
word := x -> Factorization(Gr, x);
for i in [1..Length(gens)] do for j in [i+1..Length(gens)] do
  Print("[x", i, ",x", j, "]=", word(Comm(gens[i], gens[j])), "  ");
od; od;
Print("\n");
Print("conjugation x_j^x_i as words: ");
for i in [1..Length(gens)] do for j in [1..Length(gens)] do if i <> j then
  Print("x", j, "^x", i, "=", word(gens[j]^gens[i]), "  ");
fi; od; od;
Print("\n");
Print("A = ", List(A, i -> word(els[i + 1])), "\n");
Print("B = ", List(B, i -> word(els[i + 1])), "\n");
QUIT;
