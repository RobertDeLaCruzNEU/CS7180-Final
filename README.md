1. Group members
   - Ahad Imran
   - Hamza Akmal Chaudary
   - Robert De La Cruz
2. **Problem**
Self-driving vehicles need to predict whether a pedestrian will cross the road 1-2 seconds before it happens. The cues are subtle; head turns, weight shifts, pausing at the curb. Current models infer these implicitly from raw pixels and only become accurate too late. We want to make prediction earlier and more explicit.
 
3. **Approach**
A three-stream classifier over short video clips of a pedestrian, predicting a binary label: cross or not cross.
Pose stream : Per-frame skeleton keypoints (ViTPose, pretrained). Captures gait and body orientation. Temporal transformer encoder.
Appearance stream : Cropped pedestrian frames through a pretrained backbone (ResNet-50). Captures head detail, objects, blur.
Context stream : Ego speed, distance, crosswalk presence, traffic light state. Simple MLP.
Streams are fused and classified via MLP.
 
4. **Datasets**
PIE : 6 hours, 1,842 pedestrians, rich metadata. Primary training set.
JAAD : 346 clips, 686 pedestrians. Cross-dataset evaluation.


