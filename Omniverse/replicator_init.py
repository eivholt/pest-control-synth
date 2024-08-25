import asyncio
import os
import omni.replicator.core as rep
#import omni.replicator.random as rand

rep.settings.carb_settings("/omni/replicator/RTSubframes", 1) #If randomizing materials leads to problems, try value 3

with rep.new_layer():
	def scatter_insects(insects, scatter_plane):
		with insects:
			rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (0, 360, 0)))
			rep.randomizer.scatter_2d(surface_prims=scatter_plane, check_for_collisions=True)
		return insects.node
	
	def randomize_camera(insects, camera_plane):
		with camera:
			rep.randomizer.scatter_2d(surface_prims=camera_plane, check_for_collisions=False)
			rep.modify.pose(look_at=insects)
		return camera.node
	
	def randomize_spotlights(lights):
		with lights:
			#rep.modify.attribute("intensity", rep.distribution.uniform(5000.0, 10000.0))
			#rand.intensityRange([5000.0, 10000.0])
			rep.modify.visibility(rep.distribution.choice([True, False]))
		return lights.node
	
	def randomize_distantlight(light):
		with light:
			rep.modify.attribute("intensity", rep.distribution.uniform(100.0, 2000.0))
		return light.node

	def randomize_floor(floor, materials):
		with floor:
			# Load a random scene material file as texture
			rep.randomizer.materials(materials)
		return floor.node
	
	def randomize_floor_coco(floor, texture_files):
		with floor:
			# Load a random .jpg file as texture
			rep.randomizer.texture(textures=texture_files)
		return floor.node

	rep.settings.set_render_pathtraced(samples_per_pixel=64)
	#camera = rep.create.camera(position=(0, 5, 0), projection_type="fisheyeOrthographic", focal_length=1.7)
	camera = rep.create.camera(position=(0, 5, 0))
	insects = rep.get.prims(semantics=[("class", "silverfish"), ("class", "cockroach"), ("class", "ant"), ("class", "housefly")])
	# backgrounditems = rep.get.prims(semantics=[("class", "background")])
	distant_light = rep.get.light(path_pattern='/World/DistantLight')
	spotlights = rep.get.light(semantics=[("class", "spotlight")])
	scatter_plane = rep.get.prims(path_pattern='/World/ScatterPlane')
	camera_plane = rep.get.prims(path_pattern='/World/CameraPlane')
	floor = rep.get.prims(path_pattern='/World/Floor')
	floor_materials = rep.get.material('/Looks/*')
	render_product = rep.create.render_product(camera, (640, 640))

	folder_path = 'C:/Users/eivho/source/repos/pest-control-synth/Assets/val2017/testing/'
	texture_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

	rep.randomizer.register(scatter_insects)
	rep.randomizer.register(randomize_camera)
	rep.randomizer.register(randomize_spotlights)
	rep.randomizer.register(randomize_distantlight)
	rep.randomizer.register(randomize_floor)
	rep.randomizer.register(randomize_floor_coco)
     
	with rep.trigger.on_frame(num_frames=20000, rt_subframes=20):
		rep.randomizer.scatter_insects(insects, scatter_plane)
		rep.randomizer.randomize_camera(insects, camera_plane)
		rep.randomizer.randomize_spotlights(spotlights)
		#rep.randomizer.randomize_distantlight(distant_light)
		rep.randomizer.randomize_floor(floor, floor_materials)
		#rep.randomizer.randomize_floor_coco(floor, texture_files)

	writer = rep.WriterRegistry.get("BasicWriter")
	writer.initialize(
		output_dir="C:/Users/eivho/source/repos/pest-control-synth/Dataset/out_texture_33_640",
		rgb=True,
		bounding_box_2d_loose=True)
	writer.attach([render_product])
	asyncio.ensure_future(rep.orchestrator.step_async())