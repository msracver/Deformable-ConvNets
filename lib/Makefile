all:
	cd nms/; python2 setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	cd bbox/; python2 setup_linux.py build_ext --inplace; rm -rf build; cd ../../
	cd dataset/pycocotools/; python2 setup_linux.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd nms/; rm *.so *.c *.cpp; cd ../../
	cd bbox/; rm *.so *.c *.cpp; cd ../../
	cd dataset/pycocotools/; rm *.so; cd ../../
