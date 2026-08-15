# Style profiles

A **profile** is a saved house style. It is the *look* half of the [2D
form](03-generating-references.md), stored under a name, so someone who works on two kinds of asset
does not re-pick the same settings each time they switch.

## What a profile stores

Exactly nine fields:

| Field | Why |
| --- | --- |
| base model | Which checkpoint the look depends on. |
| style LoRA | And its adapter. |
| LoRA strength | How hard the adapter is applied. |
| negative prompt | Part of the house style, not of one image. |
| platform | The prompt-side detail hint. |
| genre | A style choice. |
| era style | A style choice. |
| setting | A style choice. |
| palette | A style choice. |

What it deliberately does **not** store is the per-generation half: the prompt, the seed, the seed
lock and the candidate count are about one submit rather than about a look. Nor does it store the
taxonomy fields that describe the *subject* — category, silhouette, material, condition, emissive,
rarity and mood — which change per asset and would otherwise be dragged along by every profile
switch.

The split is not automatic. A new guidance table is far more likely to be another per-asset field
than another style one, so a field joins the profile set only by being named — which is why adding
one to `guidance.py` does not silently start dragging it between assets. See
[Extending](21-extending.md).

## Where they are managed

At the **Reference stage**, a profile picker sits beside the preset picker. Choosing one fills the fields
and makes it active, and the picker shows "Custom" once you edit past it. **Save as...** asks for a
name and captures the current form's profile fields.

**Manage...**, beside that picker, opens the full manager as a sheet over the pane. It is not a mode
and not somewhere you travel to: managing your styles is something you do to the form in front of
you, so it opens from the control it is about and Esc puts it away. (The command palette's **Manage
style profiles** is the same door from anywhere else; it goes to the Reference stage first.) **New profile**
starts from whatever the 2D form currently holds. Each saved profile lists its model, its LoRA and
how many style fields it sets, with up to four actions: **Set active**, **Edit**, **Apply to form**
and **Delete** — the active profile hides **Set active**.

Closing the sheet over a draft you have started asks before discarding it, exactly as leaving the
mode used to.

The editor works on a *draft*, not on the live form, so editing a profile never changes what your
next Generate would send. Renaming a profile in the editor moves it rather than forking it — a typo
correction does not leave you with a duplicate. Applying a profile only touches the keys the profile
actually holds, so a profile saved before a field existed leaves that new field alone rather than
blanking it.

Deleting a profile removes only the profile. Nothing already generated changes.

## The style anchor

A profile can also carry one **anchor image**: a picture every generation under that profile is
conditioned on, through the IP-Adapter. Where the nine fields above describe a look in words, an
anchor shows it — which is the difference between asking for "hand-painted texture style" and
handing the model an example of yours.

It is offered only on a profile that has been saved at least once, because the image is stored
against the profile's name.

- **Attach an anchor** opens a file picker. Any readable image works; it is copied into
  `~/.warlock/assets/profiles/` under a generated name, so it outlives every job it came from and survives a
  [prune](11-library-and-jobs.md#storage-and-pruning).
- **Strength** is how hard the anchor is applied, from 0 to 1.5. The default is 0.6.
- **Remove anchor** detaches and deletes it.

The anchor is applied through the `plus` IP-Adapter variant specifically, which conditions on 16
patch tokens rather than one pooled embedding — the difference between "the same kind of object"
and "this look", which is what an anchor is for.

Two consequences worth knowing. An anchor is not a form field, so re-saving a profile after
changing a select preserves it rather than dropping it. And attaching a conditioning reference to
one generation at the Reference stage **replaces** the profile's anchor for that asset only; the stage
says so when the active profile has one.

When ranking is on (`WARLOCK_RANK`, see [Configuration](16-configuration.md)), a finished reference
is also scored for how close it looks to the anchor. That score is advisory and appears on the
library card.
