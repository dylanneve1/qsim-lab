# Reconstruct coset codes and cover codes of Aydin, Tamo & Barg (arXiv:2606.17268)
# from their GAP index arrays, in the element numbering of export_groups.g, for
# `group_codes cparams` / `params`. Their convention: left cosets xH, a acting on
# the left, b (in N_G(H)) on the right; ours: right cosets Hx, A on the right, B on
# the left. The anti-isomorphism x -> x^-1 maps one to the other with
# A = a^-1, B = b^-1 (both printed: "inv" = mapped, "raw" = as given).
# usage: gap -q atb_codes.g   (writes <outdir>/G<order>_<id>.txt and prints one line per code)
LoadPackage("smallgrp");
outdir := "/dev/shm/qsim/qldpc-x/atb";
Export := function(G, N, id)
  local els, e, s, n, f, out, r;
  els := ShallowCopy(AsSSortedList(G));
  e := Position(els, One(G)); els[e] := els[1]; els[1] := One(G);
  s := ShallowCopy(els); n := [1..N]; SortParallel(s, n);
  f := Concatenation(outdir, "/G", String(N), "_", String(id), ".txt");
  out := OutputTextFile(f, false); SetPrintFormattingStatus(out, false);
  PrintTo(out, "G ", N, " ", id, " ", Size(Centre(G)), " 0 0 ", ReplacedString(StructureDescription(G), " ", ""), "\n");
  for r in [1..N] do
    PrintTo(out, JoinStringsWithSeparator(List([1..N], c -> String(n[PositionSorted(s, els[r]*els[c])] - 1)), " "), "\n");
  od;
  CloseStream(out);
  return x -> n[PositionSorted(s, x)] - 1;
end;
Coset := function(label, N, id, sidx, aidx, bidx)
  local G, subs, H, cosA, cosB, a, b, Nm, idx;
  G := SmallGroup(N, id);
  subs := Filtered(AllSubgroups(G), H -> not IsNormal(G, H));
  H := subs[sidx];
  cosA := LeftCosets(G, Core(G, H));
  a := List(aidx, i -> Representative(cosA[i]));
  Nm := Normalizer(G, H);
  cosB := LeftCosets(Nm, H);
  b := List(bidx, i -> Representative(cosB[i]));
  idx := Export(G, N, id);
  Print(label, " coset G=SmallGroup(", N, ",", id, ") |H|=", Size(H), " |Core|=", Size(Core(G,H)),
        " H=", JoinStringsWithSeparator(List(AsList(H), x -> String(idx(x))), ","),
        " inv: A=", JoinStringsWithSeparator(List(a, x -> String(idx(x^-1))), ","),
        " B=", JoinStringsWithSeparator(List(b, x -> String(idx(x^-1))), ","),
        " raw: A=", JoinStringsWithSeparator(List(a, x -> String(idx(x))), ","),
        " B=", JoinStringsWithSeparator(List(b, x -> String(idx(x))), ","), "\n");
end;
Cover := function(label, N, id, aidx, bidx)
  local G, els, a, b, idx;
  G := SmallGroup(N, id);
  els := Elements(G);
  a := List(aidx, i -> els[i]);
  b := List(bidx, i -> els[i]);
  idx := Export(G, N, id);
  Print(label, " 2bga G=SmallGroup(", N, ",", id, ")",
        " inv: A=", JoinStringsWithSeparator(List(a, x -> String(idx(x^-1))), ","),
        " B=", JoinStringsWithSeparator(List(b, x -> String(idx(x^-1))), ","),
        " raw: A=", JoinStringsWithSeparator(List(a, x -> String(idx(x))), ","),
        " B=", JoinStringsWithSeparator(List(b, x -> String(idx(x))), ","), "\n");
end;
# The coset-code rows index GAP 4.14.0's LeftCosets, which GAP 4.15.1 does not
# have, so only the cover (2BGA) rows, which index Elements(G), are rebuilt here.
Cover("[[168,16,10]]", 84, 13, [1, 12, 48], [1, 7, 44]);
Cover("[[280,12,16]]", 140, 9, [1, 4, 38], [1, 7, 52]);
Cover("[[112,12,8]]", 56, 8, [1, 12, 28], [1, 7, 25]);
QUIT;
