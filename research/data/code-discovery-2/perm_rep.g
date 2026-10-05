# A small faithful permutation representation of SmallGroup(N, id) and the images of the
# elements of A and B (indices in the numbering of export_groups.g), for the tests in
# tests/qec/group_codes.rs (which rebuild the group from these permutations alone).
# usage: (echo 'N:=144;; id:=167;; A:=[0,9,83];; B:=[0,51,90];;'; cat perm_rep.g) | gap -q
LoadPackage("smallgrp");
G := SmallGroup(N, id);
els := ShallowCopy(AsSSortedList(G));
e := Position(els, One(G)); els[e] := els[1]; els[1] := One(G);
iso := IsomorphismPermGroup(G);
P := Image(iso);
iso2 := SmallerDegreePermutationRepresentation(P);
f := x -> Image(iso2, Image(iso, x));
Q := Image(iso2);
deg := LargestMovedPoint(Q);
L := x -> ListPerm(f(x), deg) - 1;
Print("degree ", deg, " order ", Size(Q), "\n");
Print("gens ", List(GeneratorsOfGroup(Q), g -> ListPerm(g, deg) - 1), "\n");
Print("A ", List(A, i -> L(els[i + 1])), "\n");
Print("B ", List(B, i -> L(els[i + 1])), "\n");
QUIT;
