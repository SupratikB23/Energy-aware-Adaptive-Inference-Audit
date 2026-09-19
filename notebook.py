# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "nvidia-ml-py==13.610.43",
#     "zeus==0.16.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    def _():
        import pynvml

        print("pynvml location:", pynvml.__file__)
        print(
            "Has NVML_ERROR_MEMORY:",
            hasattr(pynvml, "NVML_ERROR_MEMORY")
        )

        pynvml.nvmlInit()

        count = pynvml.nvmlDeviceGetCount()

        print("GPU count:", count)

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            print(
                i,
                pynvml.nvmlDeviceGetName(handle)
            )
        return pynvml.nvmlShutdown()


    _()
    return


@app.cell
def _():
    def _():
        import sys
        import subprocess

        print("Python:", sys.version)

        print("\nInstalled packages:")
        subprocess.run([
            sys.executable, "-m", "pip", "list"
        ])

        print("\nNVML module:")
        import pynvml
        print("Location:", pynvml.__file__)
        print("Version:", getattr(pynvml, "__version__", "unknown"))
        print("Has _nvmlGetFunctionPointer:",
              hasattr(pynvml, "_nvmlGetFunctionPointer"))
        print("Has NVML_ERROR_MEMORY:",
              hasattr(pynvml, "NVML_ERROR_MEMORY"))

        print("\nZeus:")
        import importlib.metadata
        return print("Zeus version:",
              importlib.metadata.version("zeus"))


    _()
    return


@app.cell
def _():
    def _():
        import pynvml

        print("NVML location:", pynvml.__file__)

        print(
            "Official NVIDIA binding:",
            hasattr(pynvml, "_nvmlGetFunctionPointer")
        )

        print(
            "NVML_ERROR_MEMORY:",
            hasattr(pynvml, "NVML_ERROR_MEMORY")
        )

        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        print(
            "GPU:",
            pynvml.nvmlDeviceGetName(handle)
        )

        print(
            "Power:",
            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "W"
        )

        print(
            "Temperature:",
            pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU
            ),
            "C"
        )

        print(
            "Utilization:",
            pynvml.nvmlDeviceGetUtilizationRates(handle)
        )
        return pynvml.nvmlShutdown()


    _()
    return


@app.cell
def _():
    def _():
        import pynvml

        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        print("GPU:", pynvml.nvmlDeviceGetName(handle))
        print("GPU Utilization:", util.gpu, "%")
        print("Memory Utilization:", util.memory, "%")
        print(
            "Power:",
            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "W"
        )
        print(
            "Temperature:",
            pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU
            ),
            "°C"
        )
        return pynvml.nvmlShutdown()


    _()
    return


@app.cell
def _():
    import sys
    import subprocess

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall",
        "nvidia-ml-py==13.590.48"
    ])
    return


@app.cell
def _():
    def _():
        # ============================================================
        # 1. INSTALL
        # ============================================================

        import sys
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "zeus",
                "nvidia-ml-py",
            ],
            check=True,
        )


        # ============================================================
        # 2. IMPORTS
        # ============================================================

        import time
        import threading
        import csv

        import torch
        import torch.nn as nn
        import torch.optim as optim

        import torchvision
        import torchvision.transforms as transforms

        from torch.utils.data import DataLoader, Subset

        import pynvml


        # ============================================================
        # 3. NVML COMPATIBILITY PATCHES FOR MOLAB
        # ============================================================

        # Zeus 0.16.0 expects this constant.
        if not hasattr(pynvml, "NVML_ERROR_MEMORY"):
            pynvml.NVML_ERROR_MEMORY = 20


        # ------------------------------------------------------------
        # Zeus 0.16.0 accesses C2C information during GPU init.
        #
        # Molab's NVML Python binding does not expose:
        #
        #     nvmlDeviceGetC2cModeInfoV
        #
        # We provide a fallback that returns None.
        #
        # This is ONLY for optional C2C metadata.
        # It does NOT provide fake energy/power data.
        # ------------------------------------------------------------

        if not hasattr(pynvml, "nvmlDeviceGetC2cModeInfoV"):

            def nvmlDeviceGetC2cModeInfoV(handle):
                return None

            pynvml.nvmlDeviceGetC2cModeInfoV = (
                nvmlDeviceGetC2cModeInfoV
            )


        print("NVML_ERROR_MEMORY:",
              pynvml.NVML_ERROR_MEMORY)

        print("Official NVIDIA binding:",
              hasattr(pynvml, "_nvmlGetFunctionPointer"))

        print("C2C API available:",
              hasattr(
                  pynvml,
                  "nvmlDeviceGetC2cModeInfoV"
              ))


        # ============================================================
        # 4. INITIALIZE NVML
        # ============================================================

        pynvml.nvmlInit()

        GPU_INDEX = 0

        handle = pynvml.nvmlDeviceGetHandleByIndex(
            GPU_INDEX
        )

        gpu_name = pynvml.nvmlDeviceGetName(handle)

        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        print("\nGPU:", gpu_name)


        # ============================================================
        # 5. RAW NVML TEST
        # ============================================================

        util = pynvml.nvmlDeviceGetUtilizationRates(
            handle
        )

        power = (
            pynvml.nvmlDeviceGetPowerUsage(handle)
            / 1000.0
        )

        temperature = pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU
        )

        memory = pynvml.nvmlDeviceGetMemoryInfo(
            handle
        )

        print("\nCurrent GPU state:")
        print(
            f"GPU Utilization    : {util.gpu}%"
        )

        print(
            f"Memory Utilization : {util.memory}%"
        )

        print(
            f"Power              : {power:.2f} W"
        )

        print(
            f"Temperature        : {temperature} °C"
        )

        print(
            f"GPU Memory         : "
            f"{memory.used / (1024**2):.2f} MB / "
            f"{memory.total / (1024**2):.2f} MB"
        )


        # ============================================================
        # 6. IMPORT ZEUS
        # ============================================================

        from zeus.monitor import ZeusMonitor

        print("\n✅ Zeus imported successfully.")


        # ============================================================
        # 7. PYTORCH
        # ============================================================

        assert torch.cuda.is_available(), \
            "CUDA GPU not available."

        device = torch.device("cuda")

        print("\nPyTorch:")
        print("Device:", device)

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


        # ============================================================
        # 8. MNIST
        # ============================================================

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),
                (0.3081,)
            )
        ])


        full_train = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        full_test = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )


        # ============================================================
        # 9. SUBSET
        # ============================================================

        TRAIN_SAMPLES = 10_000
        TEST_SAMPLES = 2_000

        generator = torch.Generator().manual_seed(42)

        train_indices = torch.randperm(
            len(full_train),
            generator=generator
        )[:TRAIN_SAMPLES]

        test_indices = torch.randperm(
            len(full_test),
            generator=generator
        )[:TEST_SAMPLES]


        train_dataset = Subset(
            full_train,
            train_indices
        )

        test_dataset = Subset(
            full_test,
            test_indices
        )


        print("\nDataset:")
        print(
            "Training samples:",
            len(train_dataset)
        )

        print(
            "Test samples:",
            len(test_dataset)
        )


        # ============================================================
        # 10. DATALOADERS
        # ============================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


        # ============================================================
        # 11. RESNET-18 FOR MNIST
        # ============================================================

        model = torchvision.models.resnet18(
            num_classes=10
        )

        # Change RGB input -> grayscale
        model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove ImageNet maxpool
        model.maxpool = nn.Identity()

        model = model.to(device)


        # ============================================================
        # 12. LOSS / OPTIMIZER
        # ============================================================

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4
        )

        NUM_EPOCHS = 5

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS
        )


        # ============================================================
        # 13. GPU BASELINE
        # ============================================================

        print("\nCollecting 10-second GPU baseline...")

        baseline = []

        baseline_start = time.time()

        while time.time() - baseline_start < 10:

            try:

                util = pynvml.nvmlDeviceGetUtilizationRates(
                    handle
                )

                power = (
                    pynvml.nvmlDeviceGetPowerUsage(handle)
                    / 1000.0
                )

                temp = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )

                memory = pynvml.nvmlDeviceGetMemoryInfo(
                    handle
                )

                baseline.append({
                    "util": util.gpu,
                    "memory_util": util.memory,
                    "power": power,
                    "temperature": temp,
                    "memory_used":
                        memory.used / (1024**2)
                })

            except Exception as e:

                print(
                    "Baseline telemetry error:",
                    repr(e)
                )

            time.sleep(1)


        if baseline:

            avg_baseline_util = sum(
                x["util"] for x in baseline
            ) / len(baseline)

            avg_baseline_power = sum(
                x["power"] for x in baseline
            ) / len(baseline)

            avg_baseline_temp = sum(
                x["temperature"]
                for x in baseline
            ) / len(baseline)

        else:

            avg_baseline_util = 0
            avg_baseline_power = 0
            avg_baseline_temp = 0


        print(
            f"Baseline GPU utilization : "
            f"{avg_baseline_util:.2f}%"
        )

        print(
            f"Baseline GPU power       : "
            f"{avg_baseline_power:.2f} W"
        )

        print(
            f"Baseline temperature     : "
            f"{avg_baseline_temp:.2f} °C"
        )


        # ============================================================
        # 14. TELEMETRY
        # ============================================================

        telemetry = []

        logging = True
        experiment_start = None


        def get_process_count():

            try:

                processes = (
                    pynvml.nvmlDeviceGetComputeRunningProcesses(
                        handle
                    )
                )

                return len(processes)

            except Exception:

                return -1


        def telemetry_logger():

            while logging:

                try:

                    now = time.time()

                    util = (
                        pynvml.nvmlDeviceGetUtilizationRates(
                            handle
                        )
                    )

                    power = (
                        pynvml.nvmlDeviceGetPowerUsage(
                            handle
                        )
                        / 1000.0
                    )

                    temp = (
                        pynvml.nvmlDeviceGetTemperature(
                            handle,
                            pynvml.NVML_TEMPERATURE_GPU
                        )
                    )

                    memory = (
                        pynvml.nvmlDeviceGetMemoryInfo(
                            handle
                        )
                    )

                    gpu_clock = (
                        pynvml.nvmlDeviceGetClockInfo(
                            handle,
                            pynvml.NVML_CLOCK_GRAPHICS
                        )
                    )

                    mem_clock = (
                        pynvml.nvmlDeviceGetClockInfo(
                            handle,
                            pynvml.NVML_CLOCK_MEM
                        )
                    )

                    telemetry.append({

                        "time_sec":
                            now - experiment_start,

                        "gpu_util_percent":
                            util.gpu,

                        "memory_util_percent":
                            util.memory,

                        "power_watts":
                            power,

                        "temperature_c":
                            temp,

                        "gpu_memory_used_mb":
                            memory.used / (1024**2),

                        "gpu_memory_total_mb":
                            memory.total / (1024**2),

                        "gpu_clock_mhz":
                            gpu_clock,

                        "memory_clock_mhz":
                            mem_clock,

                        "gpu_process_count":
                            get_process_count()
                    })

                except Exception as e:

                    print(
                        "Telemetry error:",
                        repr(e)
                    )

                time.sleep(1)


        # ============================================================
        # 15. ZEUS MONITOR
        # ============================================================

        monitor = ZeusMonitor(
            gpu_indices=[GPU_INDEX]
        )

        print(
            "\n✅ ZeusMonitor created successfully."
        )


        # ============================================================
        # 16. START EXPERIMENT
        # ============================================================

        experiment_start = time.time()

        logger_thread = threading.Thread(
            target=telemetry_logger,
            daemon=True
        )

        logger_thread.start()

        torch.cuda.synchronize()

        monitor.begin_window(
            "mnist_resnet_training"
        )


        # ============================================================
        # 17. TRAINING
        # ============================================================

        print("\nStarting training...\n")

        for epoch in range(NUM_EPOCHS):

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            epoch_start = time.time()

            for images, labels in train_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += (
                    loss.item()
                    * images.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

            scheduler.step()

            epoch_time = (
                time.time() - epoch_start
            )

            epoch_loss = (
                running_loss / total
            )

            epoch_acc = (
                100.0 * correct / total
            )

            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {epoch_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )


        # ============================================================
        # 18. SYNCHRONIZE
        # ============================================================

        torch.cuda.synchronize()


        # ============================================================
        # 19. END ZEUS
        # ============================================================

        metrics = monitor.end_window(
            "mnist_resnet_training"
        )


        # ============================================================
        # 20. STOP LOGGER
        # ============================================================

        logging = False

        logger_thread.join(
            timeout=3
        )


        # ============================================================
        # 21. TEST
        # ============================================================

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()


        test_accuracy = (
            100.0 * correct / total
        )


        # ============================================================
        # 22. TELEMETRY ANALYSIS
        # ============================================================

        if telemetry:

            avg_util = sum(
                x["gpu_util_percent"]
                for x in telemetry
            ) / len(telemetry)

            avg_power = sum(
                x["power_watts"]
                for x in telemetry
            ) / len(telemetry)

            max_power = max(
                x["power_watts"]
                for x in telemetry
            )

            avg_temp = sum(
                x["temperature_c"]
                for x in telemetry
            ) / len(telemetry)

            max_temp = max(
                x["temperature_c"]
                for x in telemetry
            )

            avg_memory = sum(
                x["gpu_memory_used_mb"]
                for x in telemetry
            ) / len(telemetry)

            avg_clock = sum(
                x["gpu_clock_mhz"]
                for x in telemetry
            ) / len(telemetry)

        else:

            avg_util = 0
            avg_power = 0
            max_power = 0
            avg_temp = 0
            max_temp = 0
            avg_memory = 0
            avg_clock = 0


        # ============================================================
        # 23. SAVE CSV
        # ============================================================

        csv_file = "gpu_telemetry.csv"

        if telemetry:

            with open(
                csv_file,
                "w",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=telemetry[0].keys()
                )

                writer.writeheader()

                writer.writerows(telemetry)


        # ============================================================
        # 24. FINAL RESULTS
        # ============================================================

        print("\n")
        print("=" * 70)
        print(
            "          ZEUS + RTX PRO 6000 RESULTS"
        )
        print("=" * 70)

        print("\n--- Experiment ---")

        print(
            f"GPU               : {gpu_name}"
        )

        print(
            f"Training samples  : {TRAIN_SAMPLES}"
        )

        print(
            f"Test samples      : {TEST_SAMPLES}"
        )

        print(
            f"Epochs            : {NUM_EPOCHS}"
        )

        print(
            f"Batch size        : 128"
        )


        print("\n--- Accuracy ---")

        print(
            f"Test Accuracy     : "
            f"{test_accuracy:.2f}%"
        )


        print("\n--- Zeus ---")

        print(
            f"Training Energy   : "
            f"{metrics.total_energy:.2f} J"
        )

        print(
            f"Training Time     : "
            f"{metrics.time:.2f} sec"
        )


        print("\n--- GPU Telemetry ---")

        print(
            f"Average Util      : "
            f"{avg_util:.2f}%"
        )

        print(
            f"Average Power     : "
            f"{avg_power:.2f} W"
        )

        print(
            f"Maximum Power     : "
            f"{max_power:.2f} W"
        )

        print(
            f"Average Temp      : "
            f"{avg_temp:.2f} °C"
        )

        print(
            f"Maximum Temp      : "
            f"{max_temp:.2f} °C"
        )

        print(
            f"Average GPU Memory: "
            f"{avg_memory:.2f} MB"
        )

        print(
            f"Average GPU Clock : "
            f"{avg_clock:.2f} MHz"
        )


        print("\n--- Baseline ---")

        print(
            f"Idle Util         : "
            f"{avg_baseline_util:.2f}%"
        )

        print(
            f"Idle Power        : "
            f"{avg_baseline_power:.2f} W"
        )

        print(
            f"Idle Temperature  : "
            f"{avg_baseline_temp:.2f} °C"
        )


        print("\nTelemetry CSV:", csv_file)

        print("=" * 70)


        # ============================================================
        # 25. SHUTDOWN
        # ============================================================
        return pynvml.nvmlShutdown()


    _()
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    from torch.utils.data import DataLoader, Subset


    # ============================================================
    # 1. Device
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


    # ============================================================
    # 2. MNIST Dataset
    # ============================================================

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    full_test = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )


    # ============================================================
    # 3. Take a subset
    # ============================================================

    TRAIN_SAMPLES = 10_000
    TEST_SAMPLES = 2_000

    generator = torch.Generator().manual_seed(42)

    train_indices = torch.randperm(
        len(full_train),
        generator=generator
    )[:TRAIN_SAMPLES]

    test_indices = torch.randperm(
        len(full_test),
        generator=generator
    )[:TEST_SAMPLES]

    train_dataset = Subset(
        full_train,
        train_indices
    )

    test_dataset = Subset(
        full_test,
        test_indices
    )

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))


    # ============================================================
    # 4. DataLoaders
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    # ============================================================
    # 5. ResNet-18
    # ============================================================

    model = torchvision.models.resnet18(
        num_classes=10
    )

    # MNIST has 1 channel instead of 3
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # Remove max pooling because MNIST images are only 28x28
    model.maxpool = nn.Identity()

    model = model.to(device)


    # ============================================================
    # 6. Loss and Optimizer
    # ============================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )

    NUM_EPOCHS = 5


    # ============================================================
    # 7. Training
    # ============================================================

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_acc:.2f}%"
        )


    # ============================================================
    # 8. Evaluation
    # ============================================================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()


    test_accuracy = 100.0 * correct / total

    print("\n==============================")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("==============================")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
