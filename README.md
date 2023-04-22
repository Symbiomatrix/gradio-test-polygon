# gradio-test-polygon

- Detect polygons in sketch.
- Unique colour per numbered region, can update old ones and visually distinguish the regions.
- Draw on cumulative output image.
- Clear sketch for next region.
- Showcase mask selection.

Known issues / plans:
- Gradio won't update a sketch except emptying it when set to null (as I did). Originally the plan was for the user to receive feedback on the same image, where polygon detection is filtered to black lines only. It might've been fixed in gradio 3.23, or I'm using it wrong, I dunno.
- Currently limited to 256 regions, which is more than enough. Adding more is simple enough technically, but won't look as distinct.
- The number interface is meh, a slider would be better.
- Setting brush size to minimum doesn't work, likely due to version.
- Uploading regions mask is made possible via image defaults, this is good. Can't save 512x512 values to preset though, need to make a small "save region" button.
- Bug when checking closed contours - I can't figure out how to exclude part of the contours which is outside (for example, line shaped like "e" keeps the tail).
- Make colour gen function build on an existing list to prevent redundant calculation. And also to enable custom mask colours.
- Refactor the hell outta everything.

![RegionalPolygon1](https://user-images.githubusercontent.com/41131377/233788858-06bc4930-15e4-486a-87d6-6a8a8ae46038.png)
