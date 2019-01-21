import TensorVisualizer from '../dist';
import * as tf from '@tensorflow/tfjs';
window.tf = tf;
const Color = require('color');

export const COLORS = [
  Color.hsl([199, 100, 57]),
  Color.hsl([33, 100, 50]),
];

const attach = (id, tensor, props = {}) => {
  const root = document.getElementById(id);
  while (root.firstChild) {
    root.removeChild(root.firstChild);
  }

  new TensorVisualizer(tensor, root, {
    width: 420,
    height: 420,
    vertical: true,
    ...props,
  });
};

attach('root-1d', tf.tensor1d([ 0, 1, 2 ]));
attach('root-2d', tf.tensor2d([
  [0, 7, 4 ],
  [0, 7, 4 ],
]));
attach('root-3d', tf.tensor3d([
  [
    [0, 0],
    [0, 0],
  ],
  [
    [1, 1],
    [1, 1],
  ],
  [
    [2, 2],
    [2, 2],
  ],
  [
    [3, 3],
    [3, 3],
  ],
]));
attach('root-4d', tf.tensor4d([
  [
    [
      [0, 0],
      [0, 0],
    ],
    [
      [1, 1],
      [1, 1],
    ],
    [
      [2, 2],
      [2, 2],
    ],
    [
      [3, 3],
      [3, 3],
    ],
  ],
  [
    [
      [4, 4],
      [4, 4],
    ],
    [
      [5, 5],
      [5, 5],
    ],
    [
      [6, 6],
      [6, 6],
    ],
    [
      [7, 7],
      [7, 7],
    ],
  ],
  [
    [
      [4, 4],
      [4, 4],
    ],
    [
      [5, 5],
      [5, 5],
    ],
    [
      [6, 6],
      [6, 6],
    ],
    [
      [7, 7],
      [7, 7],
    ],
  ],
]));

attach('root-5d', tf.tensor5d([
  [
    [
      [
        [0, 0],
        [0, 0],
      ],
      [
        [1, 1],
        [1, 1],
      ],
      [
        [2, 2],
        [2, 2],
      ],
      [
        [3, 3],
        [3, 3],
      ],
    ],
    [
      [
        [4, 4],
        [4, 4],
      ],
      [
        [5, 5],
        [5, 5],
      ],
      [
        [6, 6],
        [6, 6],
      ],
      [
        [7, 7],
        [7, 7],
      ],
    ],
    [
      [
        [4, 4],
        [4, 4],
      ],
      [
        [5, 5],
        [5, 5],
      ],
      [
        [6, 6],
        [6, 6],
      ],
      [
        [7, 7],
        [7, 7],
      ],
    ],
  ],
  [
    [
      [
        [0, 0],
        [0, 0],
      ],
      [
        [1, 1],
        [1, 1],
      ],
      [
        [2, 2],
        [2, 2],
      ],
      [
        [3, 3],
        [3, 3],
      ],
    ],
    [
      [
        [4, 4],
        [4, 4],
      ],
      [
        [5, 5],
        [5, 5],
      ],
      [
        [6, 6],
        [6, 6],
      ],
      [
        [7, 7],
        [7, 7],
      ],
    ],
    [
      [
        [4, 4],
        [4, 4],
      ],
      [
        [5, 5],
        [5, 5],
      ],
      [
        [6, 6],
        [6, 6],
      ],
      [
        [7, 7],
        [7, 7],
      ],
    ],
  ],
]));
