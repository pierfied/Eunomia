all: dirs chronos likelihood

dirs:
	mkdir include && \
	mkdir lib

chronos: dirs
	git clone https://github.com/pierfied/Chronos.git && \
	cd ./Chronos && \
	pwd && \
	cmake . && \
	make && \
	mv libchronos.so ../lib && \
	mv src/*.h ../include && \
	cd .. && \
	rm -rf Chronos

likelihood: dirs chronos
	mkdir likelihood && \
	cp -r src CMakeLists.txt likelihood && \
	cd likelihood && \
	ln -s ../include include && \
	ln -s ../lib lib && \
	cmake . && \
	make && \
	mv src/*.h ../include && \
	cd .. && \
	rm -rf likelihood

clean:
	rm -rf include lib