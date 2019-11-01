---
title: Canvas中像素级别ImageData缩放
comments: true
mathjax: false
date: 2019-04-10 20:56:48
tags: [Canvas, Web, ImageData, Learning, Notes]
categories: Notes
urlname: canvas-imagedata
---

<meta name="referrer" content="no-referrer" />

{% note info %}
弄Web端手写数字识别的时候要把Canvas缩成32\*32大小的，刚开始想的是用canvas的画布缩放，弄了好久发现行不通，然后无意中发现可以获得每个像素的RGBA的值，然后又自己尝试每个像素缩放，语法不熟悉，各种尝试...
好了不多说了，代码在下面
{% endnote %}
<!--more-->

## Canvas中对ImageData数据缩放
- ctx: 原始canvas的context
- outCtx: 输出canvas的context
- scale: 缩放倍数
- scaled: 缩放后的ImageData
- imageData.data: 图像的RGBA数组，是一个一维数组

```javascript
function genImg() {
  var imgData = ctx.getImageData(0,0,500,500);
  outCtx.putImageData(scaleImageData(imgData,0.5),0,0);
}

function scaleImageData(imageData, scale) {
  var scaled =
      outCtx.createImageData(imageData.width * scale, imageData.height * scale);
  for (var row = 0; row < imageData.height; row++) {
    for (var col = 0; col < imageData.width; col++) {
      var sourcePixel = [
        imageData.data[(row * imageData.width + col) * 4 + 0],
        imageData.data[(row * imageData.width + col) * 4 + 1],
        imageData.data[(row * imageData.width + col) * 4 + 2],
        imageData.data[(row * imageData.width + col) * 4 + 3]
      ];
      for (var y = 0; y < scale; y++) {
        var destRow = Math.floor(row * scale) + y;
        for (var x = 0; x < scale; x++) {
          var destCol = Math.floor(col * scale) + x;
          for (var i = 0; i < 4; i++) {
            scaled.data[(destRow * scaled.width + destCol) * 4 + i] = sourcePixel[i];
          }
        }
      }
    }
  }
  return scaled;
}
```

{% note %}
主要就是scaleImageData这个函数，原型是stackoverflow上的，不过有bug，自己修改了下。
遇到问题果然还是要看源码...
{% endnote %}
