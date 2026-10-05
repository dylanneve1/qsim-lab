# Describe a code found by group_codes: the group SmallGroup(N, id) with the
# element numbering of export_groups.g (AsSSortedList with the identity moved
# to index 0), a small presentation, and A, B as words in its generators.
# usage: gap -q describe_code.g with N, id, A, B bound first, e.g.
#   echo 'N:=48;; id:=3;; A:=[0,5,7];; B:=[0,2,9];; Read("describe_code.g");' | gap -q
LoadPackage("smallgrp");
G := SmallGroup(N, id);
els := ShallowCopy(AsSSortedList(G));
e := Position(els, One(G)); els[e] := els[1]; els[1] := One(G);
gens := MinimalGeneratingSet(G);
iso := IsomorphismFpGroupByGenerators(G, gens);
F := Image(iso);
P := PresentationFpGroup(F);
TzGoGo(P);
Print("group: SmallGroup(", N, ",", id, ") = ", StructureDescription(G), "\n");
Print("generators (as group elements, indices): ", List(gens, x -> Position(els, x) - 1), "\n");
Print("relators: ", List(RelatorsOfFpGroup(F), String), "\n");
word := x -> String(UnderlyingElement(Image(iso, x)));
Print("A = ", List(A, i -> word(els[i + 1])), "\n");
Print("B = ", List(B, i -> word(els[i + 1])), "\n");
QUIT;
