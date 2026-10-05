# Export every group of order N (Nmin..Nmax) that is not a 2-group, not
# abelian of rank <= 2, and has a generating set of size <= 4:
# header, multiplication table (0-based, identity = 0), automorphism generators.
LoadPackage("smallgrp");
ExportOrder := function(N, dir)
  local out, i, G, inv, r, els, e, n, tab, aut, gens, s, f, sd, a;
  f := Concatenation(dir, "/", String(N), ".txt");
  out := OutputTextFile(f, false);
  # no line wrapping (GAP wraps long lines with a backslash otherwise)
  SetPrintFormattingStatus(out, false);
  for i in [1..NrSmallGroups(N)] do
    G := SmallGroup(N, i);
    if IsAbelian(G) then
      inv := AbelianInvariants(G);
      r := Maximum(List(Set(FactorsInt(N)), p -> Number(inv, q -> q mod p = 0)));
      if r <= 2 then continue; fi;
    fi;
    if Length(SmallGeneratingSet(G)) > 4 then continue; fi;
    els := ShallowCopy(AsSSortedList(G));
    e := Position(els, One(G));
    els[e] := els[1]; els[1] := One(G);
    # keep a sorted copy for lookups
    s := ShallowCopy(els);
    n := [1..N];
    SortParallel(s, n);
    tab := List([1..N], a -> List([1..N], b -> n[PositionSorted(s, els[a]*els[b])] - 1));
    aut := AutomorphismGroup(G);
    gens := GeneratorsOfGroup(aut);
    sd := StructureDescription(G);
    PrintTo(out, "G ", N, " ", i, " ", Size(Centre(G)), " ", Size(aut), " ", Length(gens), " ", ReplacedString(sd, " ", ""), "\n");
    for r in tab do PrintTo(out, JoinStringsWithSeparator(List(r, String), " "), "\n"); od;
    for a in gens do
      PrintTo(out, JoinStringsWithSeparator(List(els, x -> String(n[PositionSorted(s, Image(a, x))] - 1)), " "), "\n");
    od;
  od;
  CloseStream(out);
end;
