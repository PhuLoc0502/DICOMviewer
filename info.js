// Cấu hình Cornerstone
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWebImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.configure({
    beforeSend: function (xhr) {
        xhr.setRequestHeader('Accept', 'application/dicom');
    }
});

let currentImage = null;
let viewport = null;

// Hàm tải và hiển thị file DICOM
function loadDicom(event) {
    const file = event.target.files[0];
    if (!file) return;

    const fileReader = new FileReader();
    fileReader.onload = function (e) {
        const arrayBuffer = e.target.result;
        const byteArray = new Uint8Array(arrayBuffer);

        try {
            const dataSet = dicomParser.parseDicom(byteArray);
            displayDicomInfo(dataSet);

            const blob = new Blob([arrayBuffer], { type: "application/dicom" });
            const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(blob);

            const element = document.getElementById('dicomViewer');
            cornerstone.enable(element);

            cornerstone.loadImage(imageId).then(function (image) {
                currentImage = image;
                viewport = cornerstone.getDefaultViewportForImage(element, image);
                cornerstone.setViewport(element, viewport);
                cornerstone.displayImage(element, image);
            }).catch(function (error) {
                console.error("Lỗi tải ảnh DICOM:", error);
            });
        } catch (error) {
            console.error("Lỗi khi phân tích DICOM:", error);
        }
    };
    fileReader.readAsArrayBuffer(file);
}

// Hiển thị thông tin DICOM
function displayDicomInfo(dataSet) {
    let metadataDiv = document.getElementById("dicomMetadata");
    metadataDiv.innerHTML = `
        <p><strong>Tên Bệnh Nhân:</strong> ${getDicomValue(dataSet, 'x00100010')}</p>
        <p><strong>ID:</strong> ${getDicomValue(dataSet, 'x00100020')}</p>
        <p><strong>Ngày Chụp:</strong> ${formatDate(dataSet.string('x00080020'))}</p>
        <p><strong>Loại Hình Ảnh:</strong> ${getDicomValue(dataSet, 'x00080060')}</p>
    `;
}

// Lấy giá trị từ DICOM
function getDicomValue(dataSet, tag) {
    return dataSet.string(tag) ? dataSet.string(tag) : "Không tìm thấy";
}

// Định dạng ngày DICOM
function formatDate(dicomDate) {
    if (!dicomDate || dicomDate.length !== 8) return "Không tìm thấy";
    return `${dicomDate.slice(6, 8)}.${dicomDate.slice(4, 6)}.${dicomDate.slice(0, 4)}`;
}

// 🔥 Sửa lỗi tăng/giảm sáng
function adjustBrightness(value) {
    const element = document.getElementById('dicomViewer');
    if (!currentImage || !viewport) return;

    // Nếu Window Width không hợp lệ, gán giá trị mặc định
    if (!viewport.voi.windowWidth || viewport.voi.windowWidth <= 0) {
        viewport.voi.windowWidth = 400; // Giá trị mặc định
    }

    viewport.voi.windowCenter += value; // Điều chỉnh độ sáng bằng Window Center
    cornerstone.setViewport(element, viewport);
    cornerstone.updateImage(element);
}


// Zoom ảnh
function zoomImage(value) {
    const element = document.getElementById('dicomViewer');
    if (!currentImage || !viewport) return;

    viewport.scale += value;
    cornerstone.setViewport(element, viewport);
    cornerstone.updateImage(element);
}

// Xoay ảnh 90 độ
function rotateImage() {
    const element = document.getElementById('dicomViewer');
    if (!currentImage || !viewport) return;

    viewport.rotation += 90;
    cornerstone.setViewport(element, viewport);
    cornerstone.updateImage(element);
}


// Xử lý vẽ & tẩy
// Khởi tạo canvas cho vẽ & tẩy
// Khởi tạo canvas và trạng thái vẽ/tẩy
let isDrawing = false;
let isErasing = false;
let brushSize = 5;
let eraserSize = 10;
let canvas = document.getElementById("overlayCanvas");
let ctx = canvas.getContext("2d");
let dicomViewer = document.getElementById("dicomViewer");

// Đảm bảo canvas phủ lên ảnh DICOM
canvas.width = dicomViewer.clientWidth;
canvas.height = dicomViewer.clientHeight;
canvas.style.pointerEvents = "auto";

// Hàm bật/tắt chế độ vẽ
function toggleDrawMode() {
    if (isDrawing) {
        disableModes(); // Nếu đang bật, tắt chế độ vẽ
    } else {
        disableModes();
        isDrawing = true;
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = "red";
        ctx.lineWidth = brushSize;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        updateActiveButton("draw");
    }
}

// Hàm bật/tắt chế độ tẩy
function toggleEraseMode() {
    if (isErasing) {
        disableModes(); // Nếu đang bật, tắt chế độ tẩy
    } else {
        disableModes();
        isErasing = true;
        ctx.globalCompositeOperation = "destination-out";
        ctx.lineWidth = eraserSize;
        updateActiveButton("erase");
    }
}

// Hàm tắt tất cả chế độ
function disableModes() {
    isDrawing = false;
    isErasing = false;
    updateActiveButton("");
}

// Cập nhật kích thước bút
function updateBrushSize() {
    brushSize = document.getElementById("brushSize").value;
    if (isDrawing) ctx.lineWidth = brushSize;
}

// Cập nhật kích thước tẩy
function updateEraserSize() {
    eraserSize = document.getElementById("eraserSize").value;
    if (isErasing) ctx.lineWidth = eraserSize;
}

// Cập nhật trạng thái nút bấm
function updateActiveButton(mode) {
    document.getElementById("drawButton").classList.toggle("active", mode === "draw");
    document.getElementById("eraseButton").classList.toggle("active", mode === "erase");
}

// Xử lý vẽ/tẩy trên canvas
canvas.addEventListener("mousedown", function (event) {
    if (isDrawing || isErasing) {
        ctx.beginPath();
        ctx.moveTo(event.offsetX, event.offsetY);
        canvas.addEventListener("mousemove", draw);
    }
});

canvas.addEventListener("mouseup", function () {
    canvas.removeEventListener("mousemove", draw);
});

function draw(event) {
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
}

// Hàm lưu ảnh DICOM + nét vẽ
function saveSegmentation() {
    let tempCanvas = document.createElement("canvas");
    let tempCtx = tempCanvas.getContext("2d");

    // Đảm bảo canvas có kích thước phù hợp
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;

    // Vẽ ảnh DICOM lên canvas (nếu cornerstone đã hiển thị ảnh)
    let element = document.getElementById("dicomViewer");
    let image = cornerstone.getImage(element);
    
    if (image) {
        let viewport = cornerstone.getViewport(element);
        cornerstone.renderToCanvas(tempCanvas, image, viewport);
    } else {
        console.error("Không tìm thấy ảnh DICOM để lưu!");
        return;
    }

    // Vẽ các nét vẽ lên ảnh gốc
    tempCtx.drawImage(canvas, 0, 0);

    // Tạo link tải xuống
    let link = document.createElement("a");
    link.download = "segmentation.png";
    link.href = tempCanvas.toDataURL("image/png");
    link.click();
}

