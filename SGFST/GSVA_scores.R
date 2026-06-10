
suppressPackageStartupMessages({
  library(optparse)
  library(GSEABase)
  library(GSVA)
  library(data.table)
  library(BiocParallel)
})

opt_list <- list(
  make_option("--expr",    type = "character", help = "expression TSV"),
  make_option("--gmt",     type = "character", help = "gene-set GMT file"),
  make_option("--out",     type = "character", help = "output TSV"),
  make_option("--threads", type = "integer", default = 1, help = "CPU cores [default %default]")
)
opt <- parse_args(OptionParser(option_list = opt_list))

if (is.null(opt$expr) || is.null(opt$gmt) || is.null(opt$out)) {
  stop("Must provide --expr, --gmt and --out", call. = FALSE)
}

cat("===== GSVA script start =====\n")
cat("expr   :", opt$expr, "\n")
cat("gmt    :", opt$gmt, "\n")
cat("out    :", opt$out, "\n")
cat("threads:", opt$threads, "\n")
flush.console()

# 1. 读表达矩阵
cat("[1/5] Reading expression matrix...\n")
flush.console()
expr_dt <- fread(opt$expr, data.table = FALSE)
rownames(expr_dt) <- expr_dt[[1]]
expr_mat <- as.matrix(expr_dt[, -1, drop = FALSE])

cat("Expression matrix dim:", nrow(expr_mat), "genes x", ncol(expr_mat), "spots\n")
flush.console()

# 强制转数值矩阵，避免隐式字符/因子问题
storage.mode(expr_mat) <- "numeric"

# 2. 读 GMT
cat("[2/5] Reading GMT gene sets...\n")
flush.console()
gene_sets <- getGmt(opt$gmt)
gene_sets_list <- geneIds(gene_sets)

cat("Original gene sets:", length(gene_sets_list), "\n")
flush.console()

# 3. 过滤无效通路
cat("[3/5] Filtering gene sets by overlap...\n")
flush.console()
genes_in_expr <- rownames(expr_mat)

overlap_size <- vapply(
  gene_sets_list,
  function(gs) length(intersect(gs, genes_in_expr)),
  integer(1)
)

keep_idx <- which(overlap_size >= 10 & overlap_size <= 500)
gene_sets_list <- gene_sets_list[keep_idx]

cat("Filtered gene sets:", length(gene_sets_list), "\n")
cat("Overlap summary:\n")
print(summary(overlap_size))
flush.console()

if (length(gene_sets_list) == 0) {
  stop("No valid gene sets remain after filtering.", call. = FALSE)
}

# 4. Windows 下先强制串行，避免 SOCK 并行卡住
cat("[4/5] Running GSVA...\n")
flush.console()

bp <- SerialParam(progressbar = TRUE)

gsva_res <- NULL

# 如果输入不是原始整数 counts，而是归一化/对数化数据，建议 Gaussian
kcdf_use <- "Poisson"

try_new <- tryCatch({
  cat("Trying new GSVA API...\n")
  cat("kcdf =", kcdf_use, "\n")
  flush.console()

  gsva_param <- gsvaParam(
    exprData = expr_mat,
    geneSets = gene_sets_list,
    kcdf = kcdf_use
  )

  gsva(gsva_param, BPPARAM = bp)
}, error = function(e) {
  cat("New API failed:", conditionMessage(e), "\n")
  flush.console()
  NULL
})

if (!is.null(try_new)) {
  gsva_res <- try_new
} else {
  cat("Falling back to old GSVA API...\n")
  flush.console()

  try_old <- tryCatch({
    gsva(
      expr = expr_mat,
      gset.idx.list = gene_sets_list,
      method = "gsva",
      kcdf = kcdf_use,
      min.sz = 10,
      max.sz = 500,
      mx.diff = TRUE,
      verbose = TRUE,
      parallel.sz = 1,
      BPPARAM = bp
    )
  }, error = function(e) {
    cat("Old API failed:", conditionMessage(e), "\n")
    flush.console()
    NULL
  })

  if (is.null(try_old)) {
    stop("Both new and old GSVA API failed.", call. = FALSE)
  } else {
    gsva_res <- try_old
  }
}

cat("GSVA result dim:", nrow(gsva_res), "pathways x", ncol(gsva_res), "spots\n")
flush.console()

# 5. 输出
cat("[5/5] Writing output...\n")
flush.console()
out_dt <- as.data.frame(t(gsva_res))
fwrite(out_dt, file = opt$out, sep = "\t", quote = FALSE, row.names = TRUE)

cat("GSVA finished, result written to", opt$out, "\n")
flush.console()

invisible(gc())
closeAllConnections()