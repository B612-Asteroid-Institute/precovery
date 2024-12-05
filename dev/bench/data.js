window.BENCHMARK_DATA = {
  "lastUpdate": 1733371389850,
  "repoUrl": "https://github.com/B612-Asteroid-Institute/precovery",
  "entries": {
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "email": "akoumjian@gmail.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "akoumjian@gmail.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "distinct": true,
          "id": "fe4f4353c5a5c01359e8667f921fbb810187da43",
          "message": "Use gh pages to store benchmarks",
          "timestamp": "2024-12-04T22:23:26-05:00",
          "tree_id": "93ab62fd542e1c6cac68417e3f9eb70657fb8323",
          "url": "https://github.com/B612-Asteroid-Institute/precovery/commit/fe4f4353c5a5c01359e8667f921fbb810187da43"
        },
        "date": 1733369497138,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[1]",
            "value": 1896.4526659407416,
            "unit": "iter/sec",
            "range": "stddev: 0.00002525697561668382",
            "extra": "mean: 527.3002685273701 usec\nrounds: 1147"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[10]",
            "value": 1875.9989943656985,
            "unit": "iter/sec",
            "range": "stddev: 0.00001654230913521255",
            "extra": "mean: 533.0493262540975 usec\nrounds: 1695"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[100]",
            "value": 1677.490959976068,
            "unit": "iter/sec",
            "range": "stddev: 0.000016182119022161883",
            "extra": "mean: 596.1283988166866 usec\nrounds: 1522"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[1000]",
            "value": 762.9827550850116,
            "unit": "iter/sec",
            "range": "stddev: 0.00003782860568292412",
            "extra": "mean: 1.3106456120211785 msec\nrounds: 732"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[1]",
            "value": 4633.512567560139,
            "unit": "iter/sec",
            "range": "stddev: 0.000009309427162650748",
            "extra": "mean: 215.81898946409206 usec\nrounds: 2373"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[10]",
            "value": 3545.9124653916524,
            "unit": "iter/sec",
            "range": "stddev: 0.000010301298422745801",
            "extra": "mean: 282.01485788497837 usec\nrounds: 2118"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[100]",
            "value": 1095.5796077042937,
            "unit": "iter/sec",
            "range": "stddev: 0.000039282165064374225",
            "extra": "mean: 912.7588656888441 usec\nrounds: 886"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[1000]",
            "value": 137.91888934318416,
            "unit": "iter/sec",
            "range": "stddev: 0.0007057120602039044",
            "extra": "mean: 7.2506384351145385 msec\nrounds: 131"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[1]",
            "value": 59.329128863757155,
            "unit": "iter/sec",
            "range": "stddev: 0.00023336051671207534",
            "extra": "mean: 16.855126969694606 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[10]",
            "value": 58.36310682783065,
            "unit": "iter/sec",
            "range": "stddev: 0.00029612202040848383",
            "extra": "mean: 17.134111844833225 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[100]",
            "value": 60.5708007303558,
            "unit": "iter/sec",
            "range": "stddev: 0.0001890416939109558",
            "extra": "mean: 16.50960508928583 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[1000]",
            "value": 33.90263069923417,
            "unit": "iter/sec",
            "range": "stddev: 0.0006365558582502288",
            "extra": "mean: 29.49623611428446 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[1]",
            "value": 138.79306769751736,
            "unit": "iter/sec",
            "range": "stddev: 0.00043603620383990237",
            "extra": "mean: 7.204970799978128 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[10]",
            "value": 145.38675693665797,
            "unit": "iter/sec",
            "range": "stddev: 0.00020616219773093882",
            "extra": "mean: 6.878205560604666 msec\nrounds: 132"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[100]",
            "value": 146.98671807799082,
            "unit": "iter/sec",
            "range": "stddev: 0.00019520124898325132",
            "extra": "mean: 6.803335791669301 msec\nrounds: 144"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[1000]",
            "value": 146.22408182538274,
            "unit": "iter/sec",
            "range": "stddev: 0.000050892803755760705",
            "extra": "mean: 6.83881880136663 msec\nrounds: 146"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_precovery_search[1]",
            "value": 0.017179801519386004,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 58.20789017100003 sec\nrounds: 1"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_precovery_search[8]",
            "value": 0.01709570431437217,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 58.494226480000066 sec\nrounds: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "186861a987fa053f9e03f77b3f469bb3c1311165",
          "message": "remove unused import (#101)\n\n* remove unused import\r\n* fix alert and fail threshold",
          "timestamp": "2024-12-04T22:55:07-05:00",
          "tree_id": "ee39f4897c4b6b754c1c1a232e570330fb3823a1",
          "url": "https://github.com/B612-Asteroid-Institute/precovery/commit/186861a987fa053f9e03f77b3f469bb3c1311165"
        },
        "date": 1733371389481,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[1]",
            "value": 1881.7438844091835,
            "unit": "iter/sec",
            "range": "stddev: 0.00006592394019285944",
            "extra": "mean: 531.4219476334171 usec\nrounds: 993"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[10]",
            "value": 1875.7211826722034,
            "unit": "iter/sec",
            "range": "stddev: 0.000017304336173584222",
            "extra": "mean: 533.12827580023 usec\nrounds: 1686"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[100]",
            "value": 1689.0159847721811,
            "unit": "iter/sec",
            "range": "stddev: 0.000016607948664790992",
            "extra": "mean: 592.0607081376336 usec\nrounds: 1487"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_iterate_frame_observations[1000]",
            "value": 698.3802147504302,
            "unit": "iter/sec",
            "range": "stddev: 0.0003053065130548317",
            "extra": "mean: 1.4318847797790994 msec\nrounds: 722"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[1]",
            "value": 4611.998554428132,
            "unit": "iter/sec",
            "range": "stddev: 0.000013718775703314788",
            "extra": "mean: 216.82574012081315 usec\nrounds: 2328"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[10]",
            "value": 3483.5352727682116,
            "unit": "iter/sec",
            "range": "stddev: 0.000011774292128860936",
            "extra": "mean: 287.06469769871 usec\nrounds: 1955"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[100]",
            "value": 1083.9671299900074,
            "unit": "iter/sec",
            "range": "stddev: 0.000017885791658656492",
            "extra": "mean: 922.5371990839045 usec\nrounds: 874"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_store_observations[1000]",
            "value": 136.01115798433068,
            "unit": "iter/sec",
            "range": "stddev: 0.0003304026220764653",
            "extra": "mean: 7.352337961237019 msec\nrounds: 129"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[1]",
            "value": 56.56549456013136,
            "unit": "iter/sec",
            "range": "stddev: 0.0006942639038470176",
            "extra": "mean: 17.678622060609058 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[10]",
            "value": 57.178489960972755,
            "unit": "iter/sec",
            "range": "stddev: 0.0005344135068702953",
            "extra": "mean: 17.489094249997702 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[100]",
            "value": 58.162224156266745,
            "unit": "iter/sec",
            "range": "stddev: 0.0005567326370668048",
            "extra": "mean: 17.193290224136895 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_nbody[1000]",
            "value": 32.61398288799502,
            "unit": "iter/sec",
            "range": "stddev: 0.0004622884588080552",
            "extra": "mean: 30.661695121208055 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[1]",
            "value": 137.66967856724378,
            "unit": "iter/sec",
            "range": "stddev: 0.0005819003854382787",
            "extra": "mean: 7.263763599996764 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[10]",
            "value": 144.31501414634417,
            "unit": "iter/sec",
            "range": "stddev: 0.00015011280409026943",
            "extra": "mean: 6.929285950704612 msec\nrounds: 142"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[100]",
            "value": 144.1368343110408,
            "unit": "iter/sec",
            "range": "stddev: 0.00014370885910352108",
            "extra": "mean: 6.937851832114233 msec\nrounds: 137"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagate_orbit_2body[1000]",
            "value": 146.0398931027747,
            "unit": "iter/sec",
            "range": "stddev: 0.00007868533648652802",
            "extra": "mean: 6.84744406993133 msec\nrounds: 143"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_precovery_search[1]",
            "value": 0.016724081183542527,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 59.79401732299999 sec\nrounds: 1"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_precovery_search[8]",
            "value": 0.01651322791037668,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 60.55751216099998 sec\nrounds: 1"
          }
        ]
      }
    ]
  }
}