import bpy

pipeline = bpy.context.scene.orb.pipeline
pipeline.init(bpy.context)
while not pipeline.forward(bpy.context):
    continue
