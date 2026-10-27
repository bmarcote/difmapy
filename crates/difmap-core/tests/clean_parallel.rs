use difmap_core::clean::{clean, Window};
use difmap_core::grid::{MapBeam, MapGeom};
use rayon::ThreadPoolBuilder;

fn image() -> MapBeam {
    let nx = 1024;
    let mut map = vec![0.0; nx * nx];
    let mut beam = vec![0.0; nx * nx];
    for y in 0..nx {
        for x in 0..nx {
            let r2 = (x as f32 - 512.0).powi(2) + (y as f32 - 512.0).powi(2);
            beam[y * nx + x] = (-r2 / 20.0).exp();
            map[y * nx + x] = ((x * 17 + y * 31) % 101) as f32 / 1000.0;
        }
    }
    map[400 * nx + 600] = 2.0;
    map[600 * nx + 400] = -2.0;
    MapBeam { geom: MapGeom { nx, ny: nx, xinc: 1.0, yinc: 1.0 }, map, beam,
              e_bmaj: 1.0, e_bmin: 1.0, e_bpa: 0.0, noise: 0.0, nused: 1 }
}

#[test]
fn clean_is_bit_identical_with_one_or_four_threads() {
    for windows in [vec![], vec![Window { xmin: -200.0, xmax: 200.0, ymin: 0.0, ymax: 200.0 },
                                Window { xmin: -200.0, xmax: 200.0, ymin: -200.0, ymax: 100.0 }]] {
        for iterations in [100, -100] {
            let mut serial = image();
            let mut parallel = serial.clone();
            let a = ThreadPoolBuilder::new().num_threads(1).build().unwrap()
                .install(|| clean(&mut serial, &windows, iterations, 0.1, 0.001).unwrap());
            let b = ThreadPoolBuilder::new().num_threads(4).build().unwrap()
                .install(|| clean(&mut parallel, &windows, iterations, 0.1, 0.001).unwrap());
            assert_eq!(serial.map, parallel.map);
            assert_eq!(a.niter, b.niter);
            assert_eq!(a.cleaned_flux, b.cleaned_flux);
            assert_eq!(a.hit_cutoff, b.hit_cutoff);
            assert_eq!(a.hit_negative, b.hit_negative);
            assert_eq!(a.comps.len(), b.comps.len());
            for (a, b) in a.comps.iter().zip(&b.comps) {
                assert_eq!((a.flux, a.x, a.y), (b.flux, b.x, b.y));
            }
        }
    }
}

#[test]
fn parallel_clean_preserves_first_peak_in_window_order() {
    let mut mb = image();
    let windows = [Window { xmin: -200.0, xmax: 200.0, ymin: 0.0, ymax: 200.0 },
                   Window { xmin: -200.0, xmax: 200.0, ymin: -200.0, ymax: 0.0 }];
    let result = ThreadPoolBuilder::new().num_threads(4).build().unwrap()
        .install(|| clean(&mut mb, &windows, 1, 1.0, 0.0).unwrap());
    assert_eq!(result.comps[0].flux, -2.0);
    assert_eq!((result.comps[0].x, result.comps[0].y), (-112.0, 88.0));
}

#[test]
#[ignore]
fn clean_thread_scaling() {
    let nx = 8192;
    for threads in [1, 10] {
        let mut mb = MapBeam {
            geom: MapGeom { nx, ny: nx, xinc: 1.0, yinc: 1.0 },
            map: vec![1.0; nx * nx], beam: vec![0.0; nx * nx],
            e_bmaj: 1.0, e_bmin: 1.0, e_bpa: 0.0, noise: 0.0, nused: 1,
        };
        mb.beam[nx / 2 * nx + nx / 2] = 1.0;
        let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let start = std::time::Instant::now();
        pool.install(|| clean(&mut mb, &[], 100, 0.1, 0.0).unwrap());
        println!("{threads} threads: {:.3}s", start.elapsed().as_secs_f64());
    }
}
