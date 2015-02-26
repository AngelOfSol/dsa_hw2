#![feature(plugin)]
#![plugin(glium_macros)]
extern crate glutin;
#[macro_use]
extern crate glium;

extern crate "cgmath" as cg;


use glium::DisplayBuild;
use glium::Surface;
use glutin::Event;
use cg::*;


use std::num::Float;

#[vertex_format]
#[derive(Copy)]
struct Vertex {
    position: [f32; 3],
   
}

struct Model {
	vertices: Vec<Vertex>,
	faces: Vec<[i32; 3]>,
}

#[derive(Copy)]
struct Object3D <A> where A: std::fmt::Debug{
	center: Vector3<A>,
	axis: [Vector3<A>; 3],
	extents: [A; 3],
}

struct RenderObject {
	index_buffer: glium::IndexBuffer,
	vertices: glium::VertexBuffer<Vertex>,
}
use std::fmt;
impl<A> std::fmt::Debug for Object3D<A> where A: std::fmt::Debug {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Center({:?}, {:?}, {:?}),", self.center.x, self.center.y, self.center.z);
        for a in self.axis.iter() {
        	write!(f, "Axis({:?}, {:?}, {:?}),", a.x, a.y, a.z);
        }

        write!(f, "Extents({:?})", self.extents)

	}
}
impl<A: 'static> Object3D<A> where A: BaseFloat + std::num::Float {
	fn colliding_with(self: &Object3D<A>, other: &Object3D<A>, log: bool) -> bool {
		let test_axes = {
			let mut test_axes = vec![
				self.axis[0],
				self.axis[1],
				self.axis[2],
				other.axis[0],
				other.axis[1],
				other.axis[2],
			];
			for axis_self in self.axis.iter() {
				for axis_other in other.axis.iter() {
					if *axis_other != *axis_self {
						test_axes.push(axis_self.normalize().cross(&axis_other.normalize()).normalize());
					}
				}
			}
			test_axes
		};

		let diff = self.center - other.center;
		if log {
		println!("{:?}", self);

		println!("{:?}", other);

		println!("");
	}
		for axis in test_axes {
			let distance = diff.dot(&axis).abs();

			let self_radius = self.extents_along(&axis, log).abs();
			let other_radius = other.extents_along(&axis, log ).abs();

			if log {
				println!("{:?}", distance);
				println!("{:?}", self_radius);
				println!("{:?}", other_radius);
				println!("{:?}", axis);
				println!("LOOK HERE");
			}

			if distance > self_radius + other_radius {
				return false;
			}
		}	

		true
	}

	fn extents_along(self: &Object3D<A>, axis: &Vector3<A>, log: bool) -> A {
		let mut ret: A = Float::zero();
		let axis_n = &axis.normalize();

		let x_axis = self.axis[0].mul_s(self.extents[0]); 
		let y_axis = self.axis[1].mul_s(self.extents[1]); 
		let z_axis = self.axis[2].mul_s(self.extents[2]); 

		let one: A = Float::one();
		let two: A = one + one;

		for i in std::iter::count(-one, two).take(2) {
			for j in std::iter::count(-one, two).take(2) {
				for k in std::iter::count(-one, two).take(2) {
					let test_vec = x_axis.mul_s(i) + y_axis.mul_s(j) + z_axis.mul_s(k);
					let val = test_vec.dot(axis_n);
					ret = std::cmp::partial_max(val, ret).unwrap();
				}
			}
		}

		ret
	}

}



impl Object3D<f32> {
	fn into_model(self: &Object3D<f32>) -> Model {

		let x_axis = self.axis[0].mul_s(self.extents[0]); 
		let y_axis = self.axis[1].mul_s(self.extents[1]); 
		let z_axis = self.axis[2].mul_s(self.extents[2]); 
		
		let left_bottom_back = (-x_axis - y_axis - z_axis + self.center).into_fixed();
		let left_top_back = (-x_axis + y_axis - z_axis + self.center).into_fixed();
		let left_top_front = (-x_axis + y_axis + z_axis + self.center).into_fixed();
		let left_bottom_front = (-x_axis - y_axis + z_axis + self.center).into_fixed();

		let right_bottom_back = (x_axis - y_axis - z_axis + self.center).into_fixed();
		let right_top_back = (x_axis + y_axis - z_axis + self.center).into_fixed();
		let right_top_front = (x_axis + y_axis + z_axis + self.center).into_fixed();
		let right_bottom_front = (x_axis - y_axis + z_axis + self.center).into_fixed();

		Model {
			vertices: vec![
				Vertex { position: left_bottom_back },
				Vertex { position: left_bottom_front },
				Vertex { position: left_top_back },
				Vertex { position: left_top_front },
				Vertex { position: right_bottom_back },
				Vertex { position: right_bottom_front },
				Vertex { position: right_top_back },
				Vertex { position: right_top_front },
			],
			faces: vec![
				[0, 2, 4],
				[4, 2, 6],

				[5, 4, 7],
				[7, 4, 6],

				[2, 0, 3],
				[3, 0, 1],

				[3, 7, 2],
				[2, 7, 6],

				[7, 3, 5],
				[5, 3, 1],

				[4, 5, 0],
				[0, 5, 1],
			]
		}
	}
}

fn sign_of<A: Float>(x: A) -> A {
	if x == Float::zero() {
		Float::zero()
	} else if x > Float::zero() {
		Float::one()
	} else {
		let ret: A = Float::one();
		-ret
	}
}

impl Model {

	fn into_render_object(self: Model, display: &glium::Display) -> RenderObject {

		let mut index_list: Vec<u32> = Vec::new();
		for face in self.faces {
			for vertex in face.iter() {
				index_list.push(*vertex as u32);
			}
		} 


		let indices = glium::index::TrianglesList(index_list);

		RenderObject {
			index_buffer: glium::IndexBuffer::new(display, indices),
			vertices: glium::VertexBuffer::new(display, self.vertices),
		}
	}

	fn make_box(right: f32, top: f32, front: f32) -> Model {

		let left_bottom_back = [-right, -top, -front];
		let left_top_back = [-right, top, -front];
		let left_top_front = [-right, top, front];
		let left_bottom_front = [-right, -top, front];

		let right_bottom_back = [right, -top, -front];
		let right_top_back = [right, top, -front];
		let right_top_front = [right, top, front];
		let right_bottom_front = [right, -top, front];

		Model {
			vertices: vec![
				Vertex { position: left_bottom_back },
				Vertex { position: left_bottom_front },
				Vertex { position: left_top_back },
				Vertex { position: left_top_front },
				Vertex { position: right_bottom_back },
				Vertex { position: right_bottom_front },
				Vertex { position: right_top_back },
				Vertex { position: right_top_front },
			],
			faces: vec![
				[0, 2, 4],
				[4, 2, 6],

				[5, 4, 7],
				[7, 4, 6],

				[2, 0, 3],
				[3, 0, 1],

				[3, 7, 2],
				[2, 7, 6],

				[7, 3, 5],
				[5, 3, 1],

				[4, 5, 0],
				[0, 5, 1],
			]
		}
	}
}

struct Renderer <'a, 'b, A> where A: 'b {
	camera: Matrix4<f32>,
	perspective_mat: Matrix4<f32>,
	frame_buffer: &'b mut A,
	
	program: &'a glium::Program,
}
impl<'a, 'b, A> Renderer<'a, 'b, A> where A: glium::Surface {
	
	fn draw_model(& mut self, object: &RenderObject, mat: Matrix4<f32>, color: f32)
	{
		let uniforms = uniform! {
		    matrix:  self.perspective_mat * self.camera * mat,
		    light: [2.0, 2.0, 2.0],
		    attenuation: 0.25,
		    color: color
		};
		let mut def_render : glium::DrawParameters = std::default::Default::default();
		def_render.depth_test = glium::DepthTest::IfLess;
		def_render.depth_write = true;
		self.frame_buffer.draw(&object.vertices, &object.index_buffer, &self.program, &uniforms, &def_render).unwrap();
	}
}


trait Transformable<Transformer, Angle> {
	fn translate(self: &Self, d: &Transformer) -> Self;
	fn scale(self: &Self, d: &Transformer) -> Self;
	fn rotate(self: &Self, angle: Angle, d: &Transformer) -> Self ;
}

impl<A: 'static> Transformable<Vector3<A>, A> for Matrix4<A> where A: BaseFloat {
	fn translate(self: &Matrix4<A>, d: &Vector3<A>) -> Matrix4<A> {
		self.mul_m(&translate(d))
	}
	fn scale (self: &Matrix4<A>, d: &Vector3<A>) -> Matrix4<A> {
		self.mul_m(&scale(d))
	}
	fn rotate (self: &Matrix4<A>, angle: A, axis: &Vector3<A>) -> Matrix4<A> {
		self.mul_m(&rotate(angle, axis))
	}
}

fn translate <A: 'static>(d:&Vector3<A>) -> Matrix4<A> where A: BaseFloat  {
	Matrix4::<A>::from_translation(d)
}

fn scale <A: 'static>(d: &Vector3<A>) -> Matrix4<A> where A: BaseFloat {
	let mut ident = Matrix4::identity();
	ident[0][0] = d.x;
	ident[1][1] = d.y;
	ident[2][2] = d.z;
	ident	
}
fn rotate <A: 'static>(angle: A, axis: &Vector3<A>) -> Matrix4<A> where A: BaseFloat {
	let ret: Basis3<A> = Rotation3::from_axis_angle(axis, Rad { s: angle });
	ret.to_matrix3().to_matrix4()
}

impl<A: 'static> Transformable<Vector3<A>, A> for Object3D<A> where A: BaseFloat + Float {
	fn translate(self: &Object3D<A>, d: &Vector3<A>) -> Object3D<A> {
		Object3D {
			center: self.center + *d,
			axis: self.axis,
			extents: self.extents,
		}
	}
	fn scale (self: &Object3D<A>, d: &Vector3<A>) -> Object3D<A> {
		Object3D {
			center: self.center,
			axis: self.axis,
			extents: [
				self.extents[0] * d.x,
				self.extents[1] * d.y,
				self.extents[2] * d.z,
			]
		}
	}
	fn rotate (self: &Object3D<A>, angle: A, axis: &Vector3<A>) -> Object3D<A> {
		let rot = rotate(angle, axis);
		let t: A = Float::zero();
		Object3D {
			center: self.center,
			axis: [
				rot.mul_v(&self.axis[0].extend(t)).truncate(),
				rot.mul_v(&self.axis[1].extend(t)).truncate(),
				rot.mul_v(&self.axis[2].extend(t)).truncate(),
			],
			extents: self.extents,
		}
	}
}

fn main() {

	let obj1 = Object3D {
		center: Vector3 { x: 0.0f32, y: 0.0f32, z: 0.0f32},
		axis: [
			Vector3 { x: 1.0f32, y: 0.0f32, z: 0.0f32}.normalize(),
			Vector3 { x: 0.0f32, y: 1.0f32, z: 0.0f32}.normalize(),
			Vector3 { x: 0.0f32, y: 0.0f32, z: 1.0f32}.normalize(),
		],
		extents: [1.0f32, 1.0f32, 1.0f32],
	};

    let display = glutin::WindowBuilder::new()
        .with_dimensions(1024, 768)
        .with_title(format!("Hello world"))
        .build_glium().unwrap();
    let depth_buffer = glium::render_buffer::DepthRenderBuffer::new(&display, glium::texture::DepthFormat::I24, 1024, 768);
    let color_buffer = glium::texture::Texture2d::empty(&display, 1024, 768);
    let mut frame_buffer = glium::framebuffer::SimpleFrameBuffer::with_depth_buffer(&display, &color_buffer, &depth_buffer);

    //let box_model = Model::make_box();
    //let box_render_object = box_model.into_render_object(&display);

	let program = glium::Program::from_source(&display,
	    // vertex shader
	    "   #version 330 


	        uniform mat4 matrix;

	        attribute vec3 position;

	        varying vec3 g_pos;


	        void main() {
	            gl_Position = matrix * vec4(position, 1);
	            g_pos = position;
	        }
	    ",

	    // fragment shader
	    "   #version 330

	    	varying vec3 pos;
	    	varying vec3 norm;

	    	uniform vec3 light;
	    	uniform float attenuation;
	    	uniform float color;

	        void main() {
	        	vec3 diff = light - pos;
	        	float cos_theta = clamp(-(dot(normalize(norm), normalize(diff))), 0.0, 1.0);
	        	float dist = distance(pos, light);
	        	float inv_dist = 0.1 + cos_theta / (attenuation * dist * dist);
	            gl_FragColor = vec4(1, color, color, 1) * inv_dist;
	        }
	    ",
	    // optional geometry shader
	    Some("
	    	#version 330

	    	in vec3 g_pos[];

	    	out vec3 pos;
	    	out vec3 norm;

	    	layout(triangles) in;
	    	layout(triangle_strip, max_vertices = 3) out;

	        void main() {
	        	vec3 norm_vec = cross( g_pos[0] - g_pos[2], g_pos[0] - g_pos[1]);
	        	for(int i = 0; i < 3; i++)
	        	{
	        		gl_Position = gl_in[i].gl_Position;
	        		pos = g_pos[i];
	        		norm = norm_vec;
	        		EmitVertex();
	        	}
	        	EndPrimitive();
	        }
	    ")
	).unwrap();

	let camera = Matrix4::identity()
		.rotate(std::f32::consts::PI / 4.0f32, &Vector3::new(1.0f32, 0.0f32, 0.0f32))
		.translate(&Vector3::new(0.0f32, -3.0f32, -8.0f32))
		.rotate(std::f32::consts::PI / -4.0f32, &Vector3::new(0.0f32, 1.0f32, 0.0f32))
	;
	let size = 10.0f32;
	let perspective_mat = ortho(-size, size, -0f32, size, -size, size);//perspective(Deg { s: 45.0f32}, 1024.0f32 / 768.0f32, 0.01f32, 200.0f32);
	let mut current_rotation = 0.0f32;
	let mut done = false;	
	let mut ren = Renderer {
		camera: camera,
		perspective_mat: perspective_mat,
		frame_buffer: &mut frame_buffer,
		program: &program
	};
	let source_rect = glium::Rect {
		left: 0,
		bottom: 0,
		width: 1024,
		height: 768,
	};
	let dest_rect = glium::BlitTarget {
		left: 0,
		bottom: 0,
		width: 1024,
		height: 768,
	};
	let (mut x, mut y) = (0.0f32, 0.0f32);
	let speed = 0.03f32;
	while !done {
		let mut log = false;
		for e in display.poll_events()
		{
			match e
			{
				Event::KeyboardInput(state, _, vk) => {
					match state {
						glutin::ElementState::Pressed => match vk {
							Some(c) => {
								match c {
									glutin::VirtualKeyCode::Left => x -= speed,
									glutin::VirtualKeyCode::Right => x += speed,
									glutin::VirtualKeyCode::Up => y += speed,
									glutin::VirtualKeyCode::Down => y -= speed,
									glutin::VirtualKeyCode::L => log = true,
									_ =>(),
								}
							},
							None => (),
						},
						_ => ()
					}
					
				}
				Event::Closed => done = true,
				_ => ()
			}
		}

		current_rotation += 0.0002f32;

		let (cur_x, cur_y) = (x, y);

		fn transform_second_box<A: Transformable<Vector3<f32>, f32>>(z: A, current_rotation: f32) -> A {
			z
			.rotate(std::f32::consts::PI / 4.0f32, &Vector3::new(0.0f32, 0.0f32, 1.0f32))
			.rotate(std::f32::consts::PI / 4.0f32, &Vector3::new(1.0f32, 0.0f32, 0.0f32))
			.rotate(current_rotation, &Vector3::new(0.0f32, 1.0f32, 0.0f32))
			.translate(&Vector3::new(0.0f32, 0.0f32, 0.0f32))
		};

		let left_obj = obj1.translate(&Vector3::new(cur_x, cur_y, 0.0f32));
		let right_obj = transform_second_box(obj1, current_rotation);

		let collides = left_obj.colliding_with(&right_obj, log );

		let color = if collides {
			0.7f32
		} else {
			1.0f32
		};

		let rot = Matrix4::identity()
			//.scale(&Vector3::from_value(0.5f32))
			//.translate(&Vector3::new(cur_x, cur_y, 0.0f32))
			;
		ren.frame_buffer.clear_color(0.0, 0.0, 0.0, 1.0);
		ren.frame_buffer.clear_depth(1.0);

		ren.draw_model(&left_obj.into_model().into_render_object(&display), rot, color);

		let rot = Matrix4::identity();//transform_second_box(Matrix4::identity().scale(&Vector3::from_value(0.5f32)), current_rotation);

		ren.draw_model(&right_obj.into_model().into_render_object(&display), rot, color);


		ren.frame_buffer.blit_color(&source_rect, &display.draw(), &dest_rect, glium::uniforms::MagnifySamplerFilter::Nearest);
	}
}
/*
[
		        [ 1.0, 0.0, 0.0, 0.0 ],
		        [ 0.0, 1.0, 0.0, 0.0 ],
		        [ 0.0, 0.0, 1.0, 0.0 ],
		        [ 0.0, 0.0, 0.0, 1.0 ]
		    ]
*/