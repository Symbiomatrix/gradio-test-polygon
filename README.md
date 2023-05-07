# gradio-test-polygon

- Detect polygons in sketch.
- Unique colour per numbered region, can update old ones and visually distinguish the regions.
- Draw on cumulative output image.
- Clear sketch for next region.
- Showcase mask selection.

Known issues / plans:
- ~~Gradio won't update a sketch except emptying it when set to null (as I did). Originally the plan was for the user to receive feedback on the same image, where polygon detection is filtered to black lines only. It might've been fixed in gradio 3.23, or I'm using it wrong, I dunno.~~ Found a workaround - if the sketch's shape changes then it will refresh. Added a variant factor, and now everything is in a single sketchboard. Didn't even have to do much, using cnet's technique of image + mask, since gradio separates these to a dict. Huzzah.
- ~~Currently limited to 256 regions, which is more than enough. Adding more is simple enough technically, but won't look as distinct.~~ Increased to 360, but discovered not all colours are distinct, not sure if there's some sort of algorithm which get max distance perfectly unique in O(1), mine only does in n=2^i. For n=360 getting ~300 unique colours.
- ~~The number interface is meh, a slider would be better.~~ Done.
- Setting brush size to minimum doesn't work, likely due to version.
- ~~Uploading regions mask is made possible via image defaults, this is good. Can't save 512x512 values to preset though, need to make a small "save region" button.~~ Sorta done.
- Bug when checking closed contours - I can't figure out how to exclude part of the contours which is outside (for example, line shaped like "e" keeps the tail).
- ~~Make colour gen function build on an existing list to prevent redundant calculation. And also to enable custom mask colours.~~ Done.
- Refactor the hell outta everything. X2
- Revamped upload mechanism: Cannot override image upload directly, bug in gradio due to null mask (#4088). Instead, added a separate image for upload (buttons suck), which updates the original. Examines the image and fixes nonstandard colours to conform to the expected values, by range constraint. This is necessary since users don't understand lossy colour compression. Struggled with lossy rgb-hsv conversion issues on cv end and slowness on colorsys end, eventually struck a nice balance with nigh perfect results and high optimisation.
- Some standard colours can be uploaded which don't have an equivalent in range - they will be accepted as addenda masks outside of range cus I'm too lazy to wipe them.
- Added placeholder mask saving button. In main, should prolly be coupled with the json feature instead of standalone.
- Added mask clearing via colour -1.

![RegionalPolygon1](https://user-images.githubusercontent.com/41131377/233788858-06bc4930-15e4-486a-87d6-6a8a8ae46038.png)
