# What if I want to slow the memory down to 3us and what will happen in the modern CPU?


1. The Demysifying paper does not give any information of the ROB and MSHR which says nothing, we need to answer this question to understand the workload better.
2. If they were the first to make the delay buffer, I have some other things to ask and answer.
   1. For a serial linkage through CXL, how do we view delayed buffer? Is this remote ROB?
   2. If we have a remote ROB, how do we view the local ROB?
   3. How do we codesign the MSHR and ROB for less address translation like offloading AGU?
3. I think in the latest CPU, the ROB and MSHR are not the same thing, and the ROB is not the same thing as the delay buffer. For the workload, the decision to learn is from those two latency model.

```scala
diff --git a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/afu/afu_top.sv b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/afu/afu_top.sv
index cfcce49..22dc399 100644
--- a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/afu/afu_top.sv
+++ b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/afu/afu_top.sv
@@ -43,20 +43,132 @@ import ed_mc_axi_if_pkg::*;
       input  ed_mc_axi_if_pkg::t_to_mc_axi4    [MC_CHANNEL-1:0] cxlip2iafu_to_mc_axi4,
       output ed_mc_axi_if_pkg::t_to_mc_axi4    [MC_CHANNEL-1:0] iafu2mc_to_mc_axi4 ,
       input  ed_mc_axi_if_pkg::t_from_mc_axi4  [MC_CHANNEL-1:0] mc2iafu_from_mc_axi4,
-      output ed_mc_axi_if_pkg::t_from_mc_axi4  [MC_CHANNEL-1:0] iafu2cxlip_from_mc_axi4
-
+      output ed_mc_axi_if_pkg::t_from_mc_axi4  [MC_CHANNEL-1:0] iafu2cxlip_from_mc_axi4,
+      input logic [31:0] read_delay
 );

+localparam DW = 512+8+1+2+1;
+logic [DW-1:0] in_axi4  [MC_CHANNEL-1:0];
+logic [DW-1:0] out_axi4 [MC_CHANNEL-1:0];
+
+logic [MC_CHANNEL-1:0] in_axi4_arready;
+logic [MC_CHANNEL-1:0] out_axi4_arvalid;
+
+generate for (genvar i = 0; i < MC_CHANNEL; i = i + 1)
+begin
+  axi_r db (
+    .clock             (afu_clk),
+    .reset             (~afu_rstn),
+
+    .io_read_delay     (read_delay),
+
+    .io_axi_r_in_valid  (mc2iafu_from_mc_axi4[i].rvalid),
+    .io_axi_r_in_bits   (in_axi4[i]),
+    .io_axi_r_in_ready  (iafu2mc_to_mc_axi4[i].rready),

+    .io_axi_r_out_valid (iafu2cxlip_from_mc_axi4[i].rvalid),
+    .io_axi_r_out_bits  (out_axi4[i]),
+    .io_axi_r_out_ready (cxlip2iafu_to_mc_axi4[i].rready)
+    );
+    always @(*) begin
+      iafu2mc_to_mc_axi4[i].awaddr      = cxlip2iafu_to_mc_axi4[i].awaddr  ;
+      iafu2mc_to_mc_axi4[i].awburst     = cxlip2iafu_to_mc_axi4[i].awburst ;
+      iafu2mc_to_mc_axi4[i].awcache     = cxlip2iafu_to_mc_axi4[i].awcache ;
+      iafu2mc_to_mc_axi4[i].awid        = cxlip2iafu_to_mc_axi4[i].awid    ;
+      iafu2mc_to_mc_axi4[i].awlen       = cxlip2iafu_to_mc_axi4[i].awlen   ;
+      iafu2mc_to_mc_axi4[i].awlock      = cxlip2iafu_to_mc_axi4[i].awlock  ;
+      iafu2mc_to_mc_axi4[i].awprot      = cxlip2iafu_to_mc_axi4[i].awprot  ;
+      iafu2mc_to_mc_axi4[i].awqos       = cxlip2iafu_to_mc_axi4[i].awqos   ;
+      iafu2mc_to_mc_axi4[i].awregion    = cxlip2iafu_to_mc_axi4[i].awregion;
+      iafu2mc_to_mc_axi4[i].awsize      = cxlip2iafu_to_mc_axi4[i].awsize  ;
+      iafu2mc_to_mc_axi4[i].awuser      = cxlip2iafu_to_mc_axi4[i].awuser  ;
+      iafu2mc_to_mc_axi4[i].awvalid     = cxlip2iafu_to_mc_axi4[i].awvalid ;
+      iafu2cxlip_from_mc_axi4[i].awready= mc2iafu_from_mc_axi4[i].awready;

+      iafu2mc_to_mc_axi4[i].wdata       = cxlip2iafu_to_mc_axi4[i].wdata   ;
+      iafu2mc_to_mc_axi4[i].wlast       = cxlip2iafu_to_mc_axi4[i].wlast   ;
+      iafu2mc_to_mc_axi4[i].wstrb       = cxlip2iafu_to_mc_axi4[i].wstrb   ;
+      iafu2mc_to_mc_axi4[i].wuser       = cxlip2iafu_to_mc_axi4[i].wuser   ;
+      iafu2mc_to_mc_axi4[i].wvalid      = cxlip2iafu_to_mc_axi4[i].wvalid  ;
+      iafu2cxlip_from_mc_axi4[i].wready = mc2iafu_from_mc_axi4[i].wready ;

-//Passthrough User can implement the AFU logic here
-      assign iafu2mc_to_mc_axi4      = cxlip2iafu_to_mc_axi4;
-      assign iafu2cxlip_from_mc_axi4 = mc2iafu_from_mc_axi4;
+      iafu2cxlip_from_mc_axi4[i].bid    = mc2iafu_from_mc_axi4[i].bid    ;
+      iafu2cxlip_from_mc_axi4[i].bresp  = mc2iafu_from_mc_axi4[i].bresp  ;
+      iafu2cxlip_from_mc_axi4[i].buser  = mc2iafu_from_mc_axi4[i].buser  ;
+      iafu2cxlip_from_mc_axi4[i].bvalid = mc2iafu_from_mc_axi4[i].bvalid ;
+      iafu2mc_to_mc_axi4[i].bready      = cxlip2iafu_to_mc_axi4[i].bready  ;

+      //iafu2cxlip_from_mc_axi4[i].rdata  = mc2iafu_from_mc_axi4[i].rdata  ;
+      //iafu2cxlip_from_mc_axi4[i].rid    = mc2iafu_from_mc_axi4[i].rid    ;
+      //iafu2cxlip_from_mc_axi4[i].rlast  = mc2iafu_from_mc_axi4[i].rlast  ;
+      //iafu2cxlip_from_mc_axi4[i].rresp  = mc2iafu_from_mc_axi4[i].rresp  ;
+      //iafu2cxlip_from_mc_axi4[i].ruser  = mc2iafu_from_mc_axi4[i].ruser  ;
+      //iafu2cxlip_from_mc_axi4[i].rvalid = mc2iafu_from_mc_axi4[i].rvalid ;
+      //iafu2mc_to_mc_axi4[i].rready      = cxlip2iafu_to_mc_axi4[i].rready  ;

+      iafu2mc_to_mc_axi4[i].araddr      = cxlip2iafu_to_mc_axi4[i].araddr  ;
+      iafu2mc_to_mc_axi4[i].arburst     = cxlip2iafu_to_mc_axi4[i].arburst ;
+      iafu2mc_to_mc_axi4[i].arcache     = cxlip2iafu_to_mc_axi4[i].arcache ;
+      iafu2mc_to_mc_axi4[i].arid        = cxlip2iafu_to_mc_axi4[i].arid    ;
+      iafu2mc_to_mc_axi4[i].arlen       = cxlip2iafu_to_mc_axi4[i].arlen   ;
+      iafu2mc_to_mc_axi4[i].arlock      = cxlip2iafu_to_mc_axi4[i].arlock  ;
+      iafu2mc_to_mc_axi4[i].arprot      = cxlip2iafu_to_mc_axi4[i].arprot  ;
+      iafu2mc_to_mc_axi4[i].arqos       = cxlip2iafu_to_mc_axi4[i].arqos   ;
+      iafu2mc_to_mc_axi4[i].arregion    = cxlip2iafu_to_mc_axi4[i].arregion;
+      iafu2mc_to_mc_axi4[i].arsize      = cxlip2iafu_to_mc_axi4[i].arsize  ;
+      iafu2mc_to_mc_axi4[i].aruser      = cxlip2iafu_to_mc_axi4[i].aruser  ;
+      iafu2mc_to_mc_axi4[i].arvalid     = cxlip2iafu_to_mc_axi4[i].arvalid ;
+      iafu2cxlip_from_mc_axi4[i].arready= mc2iafu_from_mc_axi4[i].arready;

+      in_axi4[i] = {
+        mc2iafu_from_mc_axi4[i].rdata,
+        mc2iafu_from_mc_axi4[i].rid,
+        mc2iafu_from_mc_axi4[i].rlast,
+        mc2iafu_from_mc_axi4[i].rresp,
+        mc2iafu_from_mc_axi4[i].ruser
+      };
+      {
+        iafu2cxlip_from_mc_axi4[i].rdata,
+        iafu2cxlip_from_mc_axi4[i].rid,
+        iafu2cxlip_from_mc_axi4[i].rlast,
+        iafu2cxlip_from_mc_axi4[i].rresp,
+        iafu2cxlip_from_mc_axi4[i].ruser
+      } = out_axi4[i];
+    end
+end
+endgenerate
+
+`ifndef VCSSIM
+wire [255:0] dbg_in;
+assign dbg_in[0*64 +: 64]  = {
+  cxlip2iafu_to_mc_axi4[0].araddr,
+  cxlip2iafu_to_mc_axi4[0].arvalid,
+  iafu2cxlip_from_mc_axi4[0].arready,
+  cxlip2iafu_to_mc_axi4[0].arid
+  };
+
+assign dbg_in[1*64 +: 64] = {
+  iafu2mc_to_mc_axi4[0].araddr,
+  iafu2mc_to_mc_axi4[0].arvalid,
+  mc2iafu_from_mc_axi4[0].arready,
+  iafu2mc_to_mc_axi4[0].arid
+};
+
+assign dbg_in[2*64 +: 64] = {
+  mc2iafu_from_mc_axi4[0].rvalid,
+  iafu2mc_to_mc_axi4[0].rready,
+  mc2iafu_from_mc_axi4[0].rlast,
+  mc2iafu_from_mc_axi4[0].rid,
+  read_delay
+};
+
+cxldbg u0 (
+	.acq_data_in    (dbg_in), //   input,  width = 256,     tap.acq_data_in
+	.acq_trigger_in (dbg_in), //   input,  width = 256,        .acq_trigger_in
+	.acq_clk        (afu_clk) //   input,    width = 1, acq_clk.clk
+);
+`endif
 endmodule
 `ifdef QUESTA_INTEL_OEM
-`pragma questa_oem_00 "/VXyQUx++dbdtCs/nuE07Mp8M9Xqsb9R6HGCqfywzAotZHPJMKM8vRb5R+4sHFDJb5sYovyy4kKT/q0iT1sDzjJFtOZCp3cWwYEuBDWn2rvVrKj81UCDzZTWmbCzEdPghkXShZInPyq+kH3NrkrjuDdH4PaR9f8YeBeUzx1nbqLujPWnZnnaSfjmjcme6bSup+TzBjjouOUnrJqYx3WDkMUVktY+ruUuzPHiEEhBW0XFMKXPqnUsuyLHhiZbw1GhxZ6+uohVTeOm+PkUnIzWze6ZeLgjzZlWjTX+aF3wgwamZAe0hopyFIxPQ2zUsG+HCTgNymP8Cr6umyYQnezb37rgGrqxlID036y0Sg9KS8W36xCq+xTpwZa+1xOcL0BGxbpTnsaXARyl0ElSAOvvKG3sb3as2prh0n3ZhZlWQ0PRtHKFWuaKSYGC6QDfhxNBf/OOyUplnDK4lNjFUi31mrOQ6ZwiV67OmXvv/pJLPF/V7qhT2tkcMbTVfn3gC7dprOUetUmGcWKRXMtY/NKVqL7svakPaoc3odp3mmo4Q5oYyWbY9OptKP73JfenmvuMZI2MF84hVFmsiS90FsM1va6tzFPr5WfrxO5bWmb1R0OBIPI3onOQ5DAsqbjCKBZmJg0eXg5GFBDyPs7k4fj8k+mnRt3ooIyPDlxJcJ9lDUbS7c93YTqY7wxwi8xpMfUPgch5DglXG+BX2XFc3bLdGjWWArr4EXCYCVV9V8RBb4fKKK/U23intZp6zEXjqvs5w+IrhQOLhL5wUQVtAabwndnZxGPq6Xg3w/+KMkghpyx6S9QZakERXNYUUSH93cxaPUithdhQvv5ak60Jxj053+FqmHOScuJUkqxOtBEaY7sNmkNggu36m/YwrTX033XYXYwbqKefPvkYo3A1kwoLhfUCU0htShlb9qJLjgDZJw7Yz97CnLVkNXbC3fIaSn0+rn/a+ppdwfzczet7dx30UNx5QGjtHOqE4Qf+0SSm91CaguVn7mz6+EyXjVB1nDCe"
-`endif
\ No newline at end of file
+`pragma questa_oem_00 "POizRfZBu2Za2e25gOrjvm1fIPLBk0eZmyFcDIFazcJl7PX67tT/saAlNoEXLHgw5mDQeEFh0JzMQ+qx/C0+PVE6a6spr5K6BpvxdLuS075hXOTsVE7Wc/lebFBxsNWYC7WKZkRFLi9LEIJIuDzdBuFqpnd6KNxaDlPfUh7jN8WMLzEL3yixxC+CcpZ1nL96FjsMR9I8wgkeME02AXMssvm/ZFxRfH2JRVTb/5Z7jzDsL2WpgVfQCjjHJ6iHMXgtMukpclk89l2S7mS02ZKKST94bLCO2ECwg+Qx3EKTSDKbEjLPA4iRDxcG0cx9Lm6nvljWvXWNQUxcJX5cGnR3yu0fadxCvEy/bsyJ37AQeJOTGRkhql/aCDLyCb+nZtjXCNJecS5+hX0J7UXt0aPP/5Coe4GPyIL3o13OhlUy9gnw5MMa+KXm8MoygZ9Ho+GazWtkKEhqZwR9t+9defkCmebYc0ra7/3ttH5Z4Fj7vf3vDtnGK93QnK/PLVJ3ZZqVFSvV9ddXOLiBNjNdlRglX/IE8WbqJFxGUGmUnfIm7+rfGGaHeE8STkXd+Q4OWFhGPi+7+suo1KZb0vEV45VSoWGAIkdwmMewkV6KrNqUPte75hX/Az3mhdMe/xsF8Vn/6k7CsLAxiFJrRFfEEl9JGj3aUG8PTkBg9QdhrfUBCCwIuP+ru3tHaiL7/zG3HYc2K1jnmaxgtdHxGYJ+BV/bOoO0oIUw4qSNlSQiyYaJDAgkvOeAxnlWNGol76gAwWvaVQIxlA+dD7epTUECBThTwpVRQD2b+urfoi7KmamtJ0AVQY4szoiXghLRPt/jJeIYozC/3CS6PfLYQRF0DWqLy0qV5XjFpgIbp9ciRtdgFvm5T1NZfX0hHmRqDBVahGpPCtwF7CIQ3BXw4yWF5Ib0NsPQsxpbQa65b4h5e8eSmcWGeOfQHWHWs526rgAHJtGht+TVaUDB941HR7l8hkGYKJn3qrDIABM3KG4zM5hN2PRlsGy+wpiC3cWlxctnww5f"
+`endif
diff --git a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/axi_delay/build.sbt b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/axi_delay/build.sbt
new file mode 100644
index 0000000..f8cc2af
--- /dev/null
+++ b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/axi_delay/build.sbt
@@ -0,0 +1,24 @@
+// See README.md for license details.
+
+ThisBuild / scalaVersion     := "2.13.15"
+ThisBuild / version          := "0.1.0"
+ThisBuild / organization     := "%ORGANIZATION%"
+
+val chiselVersion = "6.6.0"
+
+lazy val root = (project in file("."))
+  .settings(
+    name := "%NAME%",
+    libraryDependencies ++= Seq(
+      "org.chipsalliance" %% "chisel" % chiselVersion,
+      "org.scalatest" %% "scalatest" % "3.2.16" % "test",
+    ),
+    scalacOptions ++= Seq(
+      "-language:reflectiveCalls",
+      "-deprecation",
+      "-feature",
+      "-Xcheckinit",
+      "-Ymacro-annotations",
+    ),
+    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
+  )
diff --git a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/axi_delay/src/main/scala/afu_axi_r.scala b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/axi_delay/src/main/scala/afu_axi_r.scala
new file mode 100644
index 0000000..3263d3e
--- /dev/null
+++ b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/axi_delay/src/main/scala/afu_axi_r.scala
@@ -0,0 +1,85 @@
+package afu
+
+import chisel3._
+import chisel3.util._
+import _root_.circt.stage.ChiselStage
+import chisel3.experimental.requireIsChiselType
+
+class dataBus(dw: Int, ddw: Int) extends Bundle {
+  val tmo = UInt(ddw.W)
+  val dat = UInt(dw.W)
+}
+
+class SkidBuffer[T <: Data](val gen: T) extends Module {
+  requireIsChiselType(gen)
+  val io = IO(new QueueIO(gen, 1, false))
+  io.count := 0.U
+
+  val r_ready = RegInit(true.B)
+  io.enq.ready := r_ready
+
+  val r_valid = RegInit(false.B)
+  when ((io.enq.fire) && (io.deq.valid && io.deq.ready === false.B)) {
+    r_valid := true.B
+    r_ready := false.B
+  } .elsewhen (io.deq.ready) {
+    r_valid := false.B
+    r_ready := true.B
+  }
+
+  val r_data = Reg(gen)
+  when (io.enq.fire) {
+    r_data := io.enq.bits
+  }
+
+  val o_valid = RegInit(false.B)
+  io.deq.valid := o_valid
+
+  when (io.deq.valid === false.B || io.deq.ready) {
+    o_valid := io.enq.valid || r_valid
+  }
+
+  val o_data = Reg(gen)
+  io.deq.bits := o_data
+
+  when (r_valid) {
+    o_data := r_data
+  } .elsewhen (io.enq.valid) {
+    o_data := io.enq.bits
+  }
+}
+
+class axi_r(dw: Int = 512+8+1+2+1, depth: Int = 4096, ddw: Int = 12) extends Module {
+  val io = IO(new Bundle {
+    val read_delay = Input(UInt(ddw.W))
+
+    val axi_r_in = DeqIO(UInt(dw.W))
+    val axi_r_out = EnqIO(UInt(dw.W))
+  })
+  val cycle = RegInit(0.U(ddw.W))
+  cycle := cycle + 1.U
+
+  val iskib = Module(new SkidBuffer(UInt(dw.W)))
+  io.axi_r_in <> iskib.io.enq
+
+  val rfifo = Module(new chisel3.util.Queue(new dataBus(dw, ddw), depth, useSyncReadMem = true, pipe = true))
+  rfifo.io.enq.bits.tmo := cycle + io.read_delay
+  rfifo.io.enq.bits.dat := iskib.io.deq.bits
+  rfifo.io.enq.valid := iskib.io.deq.valid
+  iskib.io.deq.ready := rfifo.io.enq.ready
+
+  val oskib = Module(new SkidBuffer(UInt(dw.W)))
+  io.axi_r_out <> oskib.io.deq
+
+  val ok = cycle === rfifo.io.deq.bits.tmo
+  oskib.io.enq.bits := rfifo.io.deq.bits.dat
+  oskib.io.enq.valid := rfifo.io.deq.valid && ok
+  rfifo.io.deq.ready := oskib.io.enq.ready && ok
+}
+
+object axi_r extends App {
+  ChiselStage.emitSystemVerilogFile(
+    new axi_r,
+    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
+  )
+}
diff --git a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_avmm_slave.sv b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_avmm_slave.sv
index 233f92c..b302782 100644
--- a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_avmm_slave.sv
+++ b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_avmm_slave.sv
@@ -40,11 +40,14 @@ module ex_default_csr_avmm_slave(
    output logic        readdatavalid,
    input  logic [31:0] address,
    input  logic        poison,
-   output logic        waitrequest
+   output logic        waitrequest,
+   output logic [31:0] read_delay
 );


  logic [31:0] csr_test_reg;
+ assign read_delay = csr_test_reg;
+
  logic [63:0] mask ;
  logic config_access;

@@ -59,19 +62,22 @@ module ex_default_csr_avmm_slave(
  assign config_access = address[21];


-//Terminating extented capability header
- localparam EX_CAP_HEADER  = 32'h00000000;
+localparam EX_CAP_HEADER  = 32'h00010023;
+localparam EX_CAP_HEADER1 = 32'h00801E98;


 //Write logic
 always @(posedge clk) begin
     if (!reset_n) begin
-        csr_test_reg <= 32'h0;
+        csr_test_reg <= 32'h20;
     end
     else begin
         if (write && (address == 22'h0000) && ~poison) begin
            csr_test_reg <= (writedata[31:0] & mask[31:0]) | (csr_test_reg & ~mask[31:0]);
         end
+        else if (write && (address[20:0] == 21'h00E08) && config_access) begin
+           csr_test_reg <= writedata & mask;
+        end
         else begin
            csr_test_reg <= csr_test_reg;
         end
@@ -90,6 +96,12 @@ always @(posedge clk) begin
         else if(read && (address[20:0] == 21'h00E00) && config_access) begin //In ED PF1 capability chain with HEADER E00 terminate here with data zero
            readdata <= {EX_CAP_HEADER} & mask;
         end
+        else if(read && (address[20:0] == 21'h00E04) && config_access) begin
+           readdata <= {EX_CAP_HEADER1} & mask;
+        end
+        else if(read && (address[20:0] == 21'h00E08) && config_access) begin
+           readdata <=  csr_test_reg & mask;
+        end
         else begin
            readdata  <= 32'h0;
         end
@@ -158,4 +170,4 @@ end
 endmodule
 `ifdef QUESTA_INTEL_OEM
 `pragma questa_oem_00 "swxC1cyBeM4mGlu/VUqBotiiOh8vVrW+2i0o+JyeZngTDWeWFUU0zpH1ZnXXqxZOPMozjYgJssVVSbEnRKrPZfDV8p+kFMdv3bzUIHvg52FXyczeOKEWkU0DTXynE36voF57lALaZI91uF1sgjq+2Loh9eTeUB22ep31OELbYMZK3Rk5QqgMYZY74zMRYeImZXKZuOJNHrbPHMl9T189w3KAuPybtvYyTZbeGIy/zU9s7gPa7wMEI3SLlfPtOcz+QbYmQtlnf55pV+Ukxh/STivqdsEbLdK96inWEZz7WgLc8XDHn4Nt8hHoLQ7OCipt176VCRZ4LAo/VXuFEE2vu+qXOX4yXdO4B1AV4yB6Q6umXSMx6vYp8FjDQWq/TbUmz5tAhJz8vvIgueL3dscM/tugaV4BMlcfg1q/xhOJhTKYkQpvO6F5seIShp9zRpKPhpAlJylXF6DRluayh3ojfVEPbIxwvC36exuKDwQCkKlNkOAJ2aq6I6vBk+EMw6dIYC6qHrmFzQvieItPp6OjM3LuhVdZ9BLsjRZRtP5QPNK2NERorcsaBjMWEilb/vEJjXMIlQtcbWtjvbg7efK4Ck/sayeWFrIv3uc3Xrw95UIfEpFnecqZO0o0LN7bpp+FAPrNShwcyBEXqrC/W3zdiOTtGKuZ5dc6FRvPfJMxDxVfs4XXU/WLK3yUKHZNdrRKNHopaRK1dIpLLdNrPKEtzMgJAuMYq23YryehZJOmKH93S5FefZotpQJXl0tSxb7Our81JtzxLKbxK1mpMjKXlyH48puNO4uEaG/5VHRO8WFWD+ush9CvALaW2gik/ea3EnIhGWyUuQv36RSFiwRMc3T5V5u7iAXFYDprhtzusQX5dYHcjocBHnyMPSOOyIMS4J+BqW4I2gHgeUneqKiOKAXDi51eSxglgipfWjScFubcgaPQvmDIq+tHsks06DhJQ+mbsPe0DQ35ABUwvDOAw16PJ/pF5VmkWwSayGNTwGvB3/1vP5fMlfMrcYxkhA2g"
-`endif
\ No newline at end of file
+`endif
diff --git a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_top.sv b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_top.sv
index 336418d..b540610 100644
--- a/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_top.sv
+++ b/intel_rtile_cxl_top_0_ed/hardware_test_design/common/ex_default_csr/ex_default_csr_top.sv
@@ -38,7 +38,8 @@ module ex_default_csr_top (
     input  logic [21:0] csr_avmm_address,
     input  logic        csr_avmm_write,
     input  logic        csr_avmm_read,
-    input  logic [7:0]  csr_avmm_byteenable
+    input  logic [7:0]  csr_avmm_byteenable,
+    output logic [31:0] read_delay
 );

 //CSR block
@@ -54,7 +55,8 @@ module ex_default_csr_top (
        .readdata     (csr_avmm_readdata),
        .readdatavalid(csr_avmm_readdatavalid),
        .address      ({10'h0,csr_avmm_address}),
-       .waitrequest  (csr_avmm_waitrequest)
+       .waitrequest  (csr_avmm_waitrequest),
+       .read_delay   (read_delay)
    );

 //USER LOGIC Implementation
@@ -65,4 +67,4 @@ module ex_default_csr_top (
 endmodule
 `ifdef QUESTA_INTEL_OEM
 `pragma questa_oem_00 "swxC1cyBeM4mGlu/VUqBotiiOh8vVrW+2i0o+JyeZngTDWeWFUU0zpH1ZnXXqxZOPMozjYgJssVVSbEnRKrPZfDV8p+kFMdv3bzUIHvg52FXyczeOKEWkU0DTXynE36voF57lALaZI91uF1sgjq+2Loh9eTeUB22ep31OELbYMZK3Rk5QqgMYZY74zMRYeImZXKZuOJNHrbPHMl9T189w3KAuPybtvYyTZbeGIy/zU+c4LjmDIztTlyPevBfOa2ZHZHGeeOPd4w/15ys+Q+orJrEPJzxQH0BHbJ6LPVxc84r0+BmRpX57AfbZ8v7V1YFfaY+EGCtwGKrWvkY8gMUsBvUveAG1bng4U5sldKMj8cgopBWCc/opCdwZXgbS++zvxe3yUOcia58MhM9h8mvguTzlZBG7LVmh/c/cqvjQHHfVDlfTA5pX8IBoFtQL9tT9i6aMIA36n49qImUHDoc5m5YMQfpjfZ5vXCJNAmkyRLXvYv67r/TigQcP0YVmSChqiz/9sGgMoTx88+tYHkGdIF5ZcD4N0E1bXb97Je+70Exy9vJJOrmncxSpAj2+0cnlhcnBzl2/VgL+XsdnSH5gPddGfLC5StrBvhz9/yxYJiRWxJB7rNexQ93IQstzvEiZI/rs24RHJS15BTsunC8bxOHui0GtRza98dCa8z0/bz+77peH8C8BaNzpjGJxY1Dy4Eg1Phjc+i+kbgUiiMvMlBh9xJds6GrpkEjL3GyKk3aaOnlS7ghdIcAWKEjUCCM34pZ7KCy7B3ZZBKNKxO6pZDsl8ktueY7LvzPy+ERCsVfQT5RPYd8N12kCfcnfJuPMc0uItGBSThXEC3Q4VN3Olhr/+/w7hzjYDkmUMOgfsYFrEqhKES+P4XKh4FzibdhIT0i04lWV0RG8UQiov9r5WvJBvvkdURCvkGzj6ph+EvLEvvGvYOH3DQ7g/jkXcZTNVpuJtCHQR70qxs6CcsRg3VbP10hCCbp5CHrAU35pguPiCiKbUJytDcreUAMgZAZ"
-`endif
\ No newline at end of file
+`endif
diff --git a/intel_rtile_cxl_top_0_ed/hardware_test_design/cxltyp2_ed.qsf b/intel_rtile_cxl_top_0_ed/hardware_test_design/cxltyp2_ed.qsf
index 29023c1..ae79cee 100644
--- a/intel_rtile_cxl_top_0_ed/hardware_test_design/cxltyp2_ed.qsf
+++ b/intel_rtile_cxl_top_0_ed/hardware_test_design/cxltyp2_ed.qsf
@@ -196,6 +196,8 @@ set_global_assignment -name SYSTEMVERILOG_FILE ./common/cafu_csr0/cafu_csr0_wrap
 set_global_assignment -name SYSTEMVERILOG_FILE ./common/cafu_csr0/cafu_csr0_avmm_wrapper.sv

 set_global_assignment -name SYSTEMVERILOG_FILE ./common/afu/afu_csr_avmm_slave.sv
+set_global_assignment -name SYSTEMVERILOG_FILE ./common/axi_delay/axi_r.sv
+set_global_assignment -name SYSTEMVERILOG_FILE ./common/axi_delay/skidbuffer.v
 set_global_assignment -name SYSTEMVERILOG_FILE ./common/afu/afu_top.sv


@@ -211,6 +213,7 @@ set_global_assignment -name IP_FILE ./common/mc_top/emif_ip/emif.ip
 set_global_assignment -name IP_FILE ./common/mc_top/emif2_ip/emif2.ip
 set_global_assignment -name IP_FILE ./common/mc_top/sip_quartus_ips/rspfifo_IP/rspfifo.ip
 set_global_assignment -name IP_FILE ./common/mc_top/sip_quartus_ips/reqfifo_IP/reqfifo.ip
+set_global_assignment -name IP_FILE ../../cxldbg.ip

 set_global_assignment -name QSYS_FILE common/intel_reset_release/intel_reset_release.ip
 set_global_assignment -name SEARCH_PATH ./common/
diff --git a/intel_rtile_cxl_top_0_ed/hardware_test_design/ed_top_wrapper_typ2.sv b/intel_rtile_cxl_top_0_ed/hardware_test_design/ed_top_wrapper_typ2.sv
index b78c78e..0b3a200 100644
--- a/intel_rtile_cxl_top_0_ed/hardware_test_design/ed_top_wrapper_typ2.sv
+++ b/intel_rtile_cxl_top_0_ed/hardware_test_design/ed_top_wrapper_typ2.sv
@@ -2221,6 +2221,7 @@ intel_cxl_tx_tlp_fifos  inst_tlp_fifos  (
   //-------------------------------------------------------
   // PF1 BAR2 example CSR                                --
   //-------------------------------------------------------
+wire [31:0] read_delay;

  ex_default_csr_top ex_default_csr_top_inst(
     .csr_avmm_clk                        ( ip2csr_avmm_clk                   ),
@@ -2233,7 +2234,8 @@ intel_cxl_tx_tlp_fifos  inst_tlp_fifos  (
     .csr_avmm_address                    ( ip2csr_avmm_address               ),
     .csr_avmm_write                      ( ip2csr_avmm_write                 ),
     .csr_avmm_read                       ( ip2csr_avmm_read                  ),
-    .csr_avmm_byteenable                 ( ip2csr_avmm_byteenable            )
+    .csr_avmm_byteenable                 ( ip2csr_avmm_byteenable            ),
+    .read_delay                          ( read_delay)
  );


@@ -2250,7 +2252,8 @@ intel_cxl_tx_tlp_fifos  inst_tlp_fifos  (
     .cxlip2iafu_to_mc_axi4            ( cxlip2iafu_to_mc_axi4    ),
     .iafu2mc_to_mc_axi4               ( iafu2mc_to_mc_axi4       ),
     .mc2iafu_from_mc_axi4             ( mc2iafu_from_mc_axi4     ),
-    .iafu2cxlip_from_mc_axi4          ( iafu2cxlip_from_mc_axi4  )
+    .iafu2cxlip_from_mc_axi4          ( iafu2cxlip_from_mc_axi4  ),
+    .read_delay                       ( read_delay               )
 );
```