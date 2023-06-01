/* Formatted on 31/05/2023 13:14:31 (QP5 v5.115.810.9015) */
SELECT   'N' AS SE_LEGADO,
         INSTRUCAO.COD AS COD,
         INSTRUCAO.NUM_ENUNCIADO AS NUM_ENUNCIADO,
         AREA.DESCR_AREA,
         AREA.COD_AREA_VCE,
         CASE INSTRUCAO.IND_QUALIDADE_FUNDAMENTO
            WHEN 'A' THEN 'ALTA'
            WHEN 'M' THEN 'MEDIA'
            ELSE 'BAIXA'
         END
            AS IND_QUALIDADE_FUNDAMENTACAO,
         INSTRUCAO.IND_PARADIGMATICO AS IND_PARADIGMATICO,
         EVENTO.DTHORA_REGISTRO AS DTHORA_ATUALIZACAO,
         acordao_sagas.descr_tipo AS tipo_processo,
         acordao_sagas.se_sigiloso AS se_acordao_sigiloso,
            acordao_sagas.num
         || '/'
         || acordao_sagas.ano
         || ' - '
         || CASE acordao_sagas.cod_colegiado
               WHEN 0 THEN 'Plenário'
               WHEN 1 THEN 'Primeira Câmara'
               WHEN 2 THEN 'Segunda Câmara'
               ELSE 'ERRO'
            END
            AS descr_norma,
         trunc(ACORDAO_SAGAS.dthora_acordao) as data_sessao, 
         INSTRUCAO.IND_TIPO_MINISTRO_AUTOR_TESE AS FUNCAO_AUTOR_TESE,
         ministro.nome_reduzido autor,
         ENUNCIADO.COD_DOCUMENTO_TRAMITAVEL AS COD_DOC_TRAMITAVEL_ENUNCIADO,
         EXCERTO.COD_DOCUMENTO_TRAMITAVEL AS COD_DOC_TRAMITAVEL_EXCERTO,
         null as descr_enunciado_legado
  FROM                     JURISPRUDENCIA.INSTRUCAO_JURISPRUDENCIA INSTRUCAO
                        INNER JOIN
                           (SELECT   area.cod AS cod_area_juris,
                                     vce.cod AS cod_area_vce,
                                     vce.termo AS descr_area
                              FROM      JURISPRUDENCIA.AREA_JURISPRUDENCIA AREA
                                     INNER JOIN
                                        APEX_CEDOC_VCE_P.vce_termo VCE
                                     ON LOWER (VCE.TERMO) =
                                           LOWER (AREA.TEXTO)
                                        AND vce.EH_DESCRITOR = 'S') area
                        ON INSTRUCAO.COD_AREA = AREA.COD_area_juris
                     -- Relacionamento para a busca do documento enunciado.
                     INNER JOIN
                        JURISPRUDENCIA.DOC_NAO_PUBLIC_INSTRUCAO_JURIS ENUNCIADO
                     ON ENUNCIADO.COD_INSTRUCAO_JURISPRUDENCIA =
                           INSTRUCAO.COD
                        AND ENUNCIADO.COD_TIPO_DOCUMENTO = 336
                  -- Relacionamento para a busca do documento excerto.
                  INNER JOIN
                     JURISPRUDENCIA.DOC_NAO_PUBLIC_INSTRUCAO_JURIS EXCERTO
                  ON EXCERTO.COD_INSTRUCAO_JURISPRUDENCIA = INSTRUCAO.COD
                     AND EXCERTO.COD_TIPO_DOCUMENTO = 200
               -- Recupera a data da ultima oficialização.
               INNER JOIN
                  (SELECT   DISTINCT
                            COD_INSTRUCAO_JURISPRUDENCIA,
                            MAX(DTHORA_REGISTRO)
                               OVER (
                                  PARTITION BY COD_INSTRUCAO_JURISPRUDENCIA
                               )
                               DTHORA_REGISTRO
                     FROM   JURISPRUDENCIA.EVENTO_INSTRUCAO_JURIS
                    WHERE   COD_TIPO_EVENTO = 19) EVENTO
               ON INSTRUCAO.COD = EVENTO.COD_INSTRUCAO_JURISPRUDENCIA
            INNER JOIN
               JURISPRUDENCIA.VW_ACORDAO_UNITARIO_OFIC_JUSIS ACORDAO_SAGAS
            ON INSTRUCAO.COD_ACORDAO_UNITARIO = ACORDAO_SAGAS.COD
         INNER JOIN
            tcu.ministro ministro
         ON INSTRUCAO.COD_MINISTRO_AUTOR_TESE = ministro.cod
 WHERE   INSTRUCAO.COD_SITUACAO_ATUAL IN (15, 16) -- OFICIALIZADO OU PUBLICADO
UNION ALL
-- dados legados: count(*) = 227
SELECT   'S' AS SE_LEGADO,
         legado.COD AS COD,
         NULL AS NUM_ENUNCIADO,
         AREA.DESCR_AREA,
         AREA.COD_AREA_VCE,
         CASE IND_QUALIDADE_FUNDAMENTO
            WHEN 'A' THEN 'ALTA'
            WHEN 'M' THEN 'MEDIA'
            ELSE 'BAIXA'
         END
            AS IND_QUALIDADE_FUNDAMENTACAO,
         'SUMULA' AS IND_PARADIGMATICO,
         NULL AS DTHORA_ATUALIZACAO,
         NULL AS tipo_processo,
         NULL AS se_acordao_sigiloso,
         descr_aprovacao AS descr_norma,
         trunc(legado.data_sessao) as data_sessao, 
         'RELATOR' AS FUNCAO_AUTOR_TESE,
         ministro.nome_reduzido autor,
         NULL AS COD_DOC_TRAMITAVEL_ENUNCIADO,
         NULL AS COD_DOC_TRAMITAVEL_EXCERTO,
         descr_enunciado as descr_enunciado_legado
  FROM         jurisprudencia.sumula_antiga legado
            INNER JOIN
               (SELECT   area.cod AS cod_area_juris,
                         vce.cod AS cod_area_vce,
                         vce.termo AS descr_area
                  FROM      JURISPRUDENCIA.AREA_JURISPRUDENCIA AREA
                         INNER JOIN
                            APEX_CEDOC_VCE_P.vce_termo VCE
                         ON LOWER (VCE.TERMO) = LOWER (AREA.TEXTO)
                            AND vce.EH_DESCRITOR = 'S') area
            ON legado.COD_AREA = AREA.COD_area_juris
         INNER JOIN
            tcu.ministro ministro
         ON legado.cod_relator = ministro.cod
