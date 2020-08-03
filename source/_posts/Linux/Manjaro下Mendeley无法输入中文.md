---
title: Manjaro下Mendeley无法输入中文
tags:
  - Linux
  - Manjaro
  - Arch
  - Mendeley
categories:
  - Linux
comments: true
mathjax: false
date: 2020-07-30 16:12:40
urlname: manjaro-mendeley-fcitx
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言

Mendeley是一款论文管理软件，可以跨平台同步使用。  

在Manjaro上发现Mendeley无法输入中文，进行批注。  
查找原因是Mendeley使用了自己的QT库，没有对fcitx的支持。
{% endnote %}
<!--more-->

## 解决办法

网上查到了基本上都是添加`/opt/mendeleydesktop/lib/mendeleydesktop/plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so`  

方法大同小异，可以参考[这个博客](https://www.findhao.net/easycoding/2287)有的是从github上自己下载放里面，有的是用`locate libfcitxplatforminputcontextplugin.so`后进行本地的软链接。但是这些方法在我这都失效了。

后续发现[一个帖子](https://www.zhihu.com/question/25517900/answer/236861798)，说是Mendeley版本升级后，对Qt的依赖版本也升级了，要下载对应的版本。Mendeley 1.18-1.191使用了Qt5.10.1的库，但是进行操作后还是无法输入中文。。

最后在AUR中搜到了一个[AUR : mendeleydesktop-bundled-fcitx.git](https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=mendeleydesktop-bundled-fcitx)，尝试了下，ok了。

查看了它的PKGBUILD，发现安装的不止上面方面说的那个库，还包括了`/opt/mendeleydesktop/lib/qt/libFcitxQt5DBusAddons.so.1.0`这个包，可以从[github](https://github.com/yinflying/BlogSource/raw/master/lib-fcitx-plugin/arch-qt5.10.1/libFcitxQt5DBusAddons.so.1.0)上下载。


## PKGBUILD

具体的PKGBUILD如下：

```
# This is an example PKGBUILD file. Use this as a start to creating your own,
# and remove these comments. For more information, see 'man PKGBUILD'.
# NOTE: Please fill out the license field for your package! If it is unknown,
# then please put 'unknown'.

# The following guidelines are specific to BZR, GIT, HG and SVN packages.
# Other VCS sources are not natively supported by makepkg yet.

# Maintainer: yinflying <yinflying@foxmail.com>
pkgname=mendeleydesktop-bundled-fcitx
pkgver=1.19.4
pkgrel=1
pkgdesc="mendeleydesktop fcitx input method support"
arch=('x86_64')
url="http://yinflying.top/2017/09/727"
license=('GPL')
depends=('mendeleydesktop-bundled')
makedepends=('git')  # 'bzr', 'git', 'mercurial' or 'subversion'
provides=("${pkgname}")
conflicts=("${pkgname}")
replaces=()
backup=()
options=()
install=
source=("https://github.com/yinflying/BlogSource/raw/master/lib-fcitx-plugin/arch-qt5.10.1/libFcitxQt5DBusAddons.so.1.0"
        "https://github.com/yinflying/BlogSource/raw/master/lib-fcitx-plugin/arch-qt5.10.1/libfcitxplatforminputcontextplugin.so")
noextract=()
md5sums=('SKIP' 'SKIP')

package() {
    install -Dm755 "$srcdir/libfcitxplatforminputcontextplugin.so" "$pkgdir/opt/mendeleydesktop/lib/mendeleydesktop/plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so"
    install -Dm755 "$srcdir/libFcitxQt5DBusAddons.so.1.0" "$pkgdir/opt/mendeleydesktop/lib/qt/libFcitxQt5DBusAddons.so.1.0"
    cd "$pkgdir/opt/mendeleydesktop/lib/qt"
    ln -s libFcitxQt5DBusAddons.so.1.0 libFcitxQt5DBusAddons.so.1
    ln -s libFcitxQt5DBusAddons.so.1 libFcitxQt5DBusAddons.so
}
```

----------
