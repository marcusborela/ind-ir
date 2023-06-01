/* Formatted on 31/05/2023 14:18:44 (QP5 v5.115.810.9015) */
SELECT   'S' AS SE_LEGADO,
         sumula.cod AS cod,
         INDEXA.NUM_NIVEL,
         vce_juris.texto,
         vce_juris.cod AS cod_termo_juris
  FROM         JURISPRUDENCIA.SUMULA_ANTIGA SUMULA
            INNER JOIN
               jurisprudencia.SUMULA_TEXTO_INDEXADO indexa
            ON SUMULA.cod = INDEXA.COD_SUMULA
         INNER JOIN
            jurisprudencia.texto_indexacao_juris vce_juris
         ON INDEXA.COD_TEXTO_INDEXACAO = vce_juris.cod
UNION ALL
SELECT   'N' AS SE_LEGADO,
         INSTRUCAO.cod AS cod,
         INDEXA.NUM_NIVEL,
         vce_juris.texto,
         vce_juris.cod AS cod_termo_juris
  FROM         JURISPRUDENCIA.INSTRUCAO_JURISPRUDENCIA INSTRUCAO
            INNER JOIN
               jurisprudencia.INSTRUCAO_TEXTO_INDEXADO_JURIS indexa
            ON instrucao.cod = INDEXA.COD_INSTRUCAO_JURISPRUDENCIA
         INNER JOIN
            jurisprudencia.texto_indexacao_juris vce_juris
         ON INDEXA.COD_TEXTO_INDEXACAO = vce_juris.cod
 WHERE   instrucao.cod_situacao_atual IN (15, 16)
 