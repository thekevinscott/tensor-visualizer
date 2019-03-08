import rough from 'roughjs';
import * as tf from '@tensorflow/tfjs';

import {
  IProps,
} from './types';

const RECT_SIZE = 80;
const FONT_SIZE = 20;
const FONT_FAMILY = 'Sofia Pro';
const FONT_PADDING = 0;
const ROUGHNESS = 0.7;
const PADDING = 10;
const Z_PADDING = 10;

type ITensor = tf.Tensor;

type IRenderTensorProps = {
  x: number;
  y: number;
  width: number;
  height: number;
}

// const getRank = (tensor: ITensor, depth = 1): number => {
//   if (typeof tensor[0] === 'number') {
//     return depth;
//   }

//   return getRank(tensor[0], depth + 1);
// }

class TensorVisualizer {
  private canvas: HTMLCanvasElement;
  private rc: any;
  private roughness: number;
  private fontSize: number;
  private fontPadding: number;
  private vertical: boolean;
  private width: number;
  private height: number;
  private zPadding: number;

  constructor(tensor: tf.Tensor, target: HTMLElement, props:IProps) {
    const {
      width,
      height,
      roughness,
      fontSize,
      fontFamily,
      fontPadding,
      vertical,
      zPadding,
    } = props;

    this.width = width;
    this.height = height;
    this.zPadding = zPadding === undefined ? Z_PADDING : zPadding;
    this.vertical = vertical || false;
    this.roughness = roughness === undefined ? ROUGHNESS : roughness;
    this.fontSize = fontSize === undefined ? FONT_SIZE : fontSize;
    this.fontPadding = fontPadding === undefined ? FONT_PADDING : fontPadding;
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    const ctx = this.canvas.getContext("2d");
    ctx.font = `${this.fontSize}px ${fontFamily || FONT_FAMILY}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';

    const deviceScaleFactor = props.deviceScaleFactor || 1;
    this.canvas.style.width = `${width / deviceScaleFactor}px`;
    this.canvas.style.height = `${height / deviceScaleFactor}px`;
    this.canvas.getContext('2d').scale(deviceScaleFactor, deviceScaleFactor);
    target.appendChild(this.canvas);
    this.rc = rough.canvas(this.canvas);

    this.render(tensor);
  }

  render = (tensor: tf.Tensor) => {
    // const padding = PADDING;
    // this.renderTensor(
      // [0, 7, 4, 5, 5],
    // {
    this.renderTensor(tensor, {
      x: PADDING,
      y: PADDING,
      width: this.width - (PADDING * 2),
      height: this.height - (PADDING * 2),
    });
  }

  renderTensor = (tensor: ITensor, props: IRenderTensorProps) => {
    const rank = Number(tensor.rankType);

    if (tensor.shape.length === 1) {
      return this.render1DTensor(tensor, props);
    } else if (tensor.shape.length === 2) {
      return this.render2DTensor(tensor, props);
    } else if (tensor.shape.length === 3) {
      return this.render3DTensor(tensor, props);
    } else if (tensor.shape.length === 4) {
      return this.render4DTensor(tensor, props);
    } else if (tensor.shape.length === 5) {
      return this.render5DTensor(tensor, props);
    }
  }

  render1DTensor = (tensor: tf.Tensor, {
    x,
    y,
    width,
    height,
  }: IRenderTensorProps) => {
    for (let i = 0; i < tensor.shape[0]; i++) {
      const num = tensor.dataSync()[i];
      if (this.vertical) {
        const rectSize = height / tensor.shape[0];
        this.renderRect(num, x, y + (rectSize * i), width, rectSize);
      } else {
        const rectSize = width / tensor.shape[0];
        this.renderRect(num, x + (rectSize * i), y, rectSize, height);
      }
    }
  }

  render2DTensor = (tensor: tf.Tensor, {
    x,
    y,
    width,
    height,
  }: IRenderTensorProps) => {
    const shape = tensor.shape[0];
    for (let i = 0; i < tensor.shape[0]; i++) {
      const sectionSize = height / shape;

      if (this.vertical) {
        const rectSize = width / shape;
        this.render1DTensor(tensor.slice(i, 1).squeeze(), {
          x: x + (rectSize * i),
          y,
          width: width / shape,
          height,
        });
      } else {
        const rectSize = height / shape;
        this.render1DTensor(tensor.slice(i, 1).squeeze(), {
          x,
          y: y + (rectSize * i),
          width,
          height: height / shape,
        });
      }
    }
  }

  render3DTensor = (tensor: tf.Tensor, props: IRenderTensorProps) => {
    const len = tensor.shape[0] - 1;
    for (let i = len; i >= 0; i--) {
      const sizePadding = this.zPadding * len;
      this.render2DTensor(tensor.slice(i, 1).squeeze(), {
        ...props,
        x: props.x + (this.zPadding * i),
        y: props.y + (this.zPadding * (len - i)),
        width: props.width - sizePadding,
        height: props.height - sizePadding,
      });
    }

  }

  render4DTensor = (tensor: tf.Tensor, {
    x,
    y,
    height,
    width,
  }: IRenderTensorProps) => {
    const len = tensor.shape[0] - 1;
    const padding = -10;
    const totalPadding = padding * (tensor.shape[0] - 1);
    const shape = tensor.shape[0];
    if (this.vertical) {
      height += padding;
      for (let i = shape - 1; i >= 0; i--) {
        const rectSize = (height / shape) - (padding);
        this.render3DTensor(tensor.slice(i, 1).squeeze(), {
          x,
          y: y + (rectSize * i) + (padding * i),
          width,
          height: rectSize,
        });
      }
    } else {
      width += padding;
      for (let i = 0; i < shape; i++) {
        const rectSize = (width / shape) - padding;
        this.render3DTensor(tensor.slice(i, 1).squeeze(), {
          x: x + (rectSize * i) + (padding * i),
          y,
          width: rectSize,
          height,
        });
      }
    }
  }

  render5DTensor = (tensor: tf.Tensor, {
    x,
    y,
    height,
    width,
  }: IRenderTensorProps) => {
    const len = tensor.shape[0] - 1;
    const padding = 10;
    const shape = tensor.shape[0];
    const totalPadding = padding * (shape - 1);
    if (this.vertical) {
      width += padding;
      for (let i = 0; i < shape; i++) {

        const rectSize = (width / shape) - padding;
        this.render4DTensor(tensor.slice(i, 1).squeeze(), {
          x: x + (rectSize * i),
          y,
          width: width / shape,
          height,
        });
      }
    } else {
      for (let i = shape - 1; i >= 0; i--) {
        const rectSize = (height / shape) - padding;
        this.render4DTensor(tensor.slice(i, 1).squeeze(), {
          x,
          y: y + (rectSize * i),
          width,
          height: height / shape,
        });
      }
    }
  }

  renderRect = (num: number, x: number, y: number, width: number = RECT_SIZE, height: number = RECT_SIZE) => {
    this.rc.rectangle(x, y, width, height, {
      roughness: this.roughness,
      fill: 'rgba(242, 248, 252, 0.95)',
      fillStyle: 'solid',
    });
    const ctx = this.canvas.getContext('2d');
    ctx.fillText(`${num}`, x + width / 2, this.fontPadding + y + height + ((this.fontSize - height) / 2));
  }
}

export default TensorVisualizer;
