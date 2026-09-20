// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The verifier's verdicts, in words — ONE list for the label pill, the
// object card and the report, so a new verdict cannot get a colour in one
// place and no mark in another (they disagreed by accident before the shared table).
// The pill carries a MARK: "?" when the LABEL is in doubt, "≈" when the
// label stood but the export could not say whether it is one object.
// REJECT never reaches a scene: rejected instances are removed.

export const VERDICTS = {
  CONFIRM: { mark: "", tone: "ok",
             word: "confirmed by the verifier" },
  RELABEL: { mark: "", tone: "ok",
             word: "relabelled by the verifier" },
  RELABEL_UNSUPPORTED: {
    mark: "?", tone: "warn",
    word: "label kept; the verifier suggested another name, but too few "
          + "views agreed" },
  RELABEL_OUTSIDE_VOCAB: {
    mark: "?", tone: "warn",
    word: "label kept; the verifier suggested a name that is not in this "
          + "scene's vocabulary" },
  RELABEL_TOOLARGE: {
    mark: "?", tone: "warn",
    word: "label kept; the verifier suggested another name, but the object "
          + "fills too much of the frame to check it" },
  HELD: { mark: "≈", tone: "warn",
          word: "held; the label stood, but this may be more than one "
                + "object, or part of one" },
  UNVERIFIED: { mark: "?", tone: "warn",
                word: "unverified; the verifier's answer could not be read" },
  NOT_JUDGED: { mark: "?", tone: "warn",
                word: "not judged: not marked for verification; the label "
                      + "ships as detected" },
};

/** {mark, tone, word} for a verdict string; an unknown verdict is shown
 *  as doubt rather than hidden. Empty = not verified (no verdict yet). */
export function verdictInfo(verdict) {
  const v = String(verdict ?? "");
  if (!v) return { mark: "", tone: "none", word: "" };
  return VERDICTS[v] ?? { mark: "?", tone: "warn", word: `verifier: ${v}` };
}
