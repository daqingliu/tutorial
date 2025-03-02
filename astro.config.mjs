// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkMdxImages from 'remark-mdx-images';
import { remarkModifiedTime } from './remark-modified-time.mjs';
import mdx from '@astrojs/mdx';


// https://astro.build/config
export default defineConfig({
    site: 'https://daqingliu.github.io',
    base: 'tutorial',
    integrations: [starlight({
        title: 'AIGC From Scratch',
		defaultLocale: 'zh-CN',
        lastUpdated: true,
        customCss: [
			'./src/styles/custom.css',
            "/node_modules/katex/dist/katex.min.css",
          ],
        social: {
            github: 'https://github.com/daqingliu',
        },
        sidebar: [
            {
                label: 'Diffusion Model',
                autogenerate: { directory: 'diffusion-model' },
            },
            {
                label: 'Diffusers',
                autogenerate: { directory: 'diffusers' },
            },
            {
                label: 'ComfyUI',
                autogenerate: { directory: 'comfyui' },
            },
        ],
		}),
        mdx()],
    markdown: {
        remarkPlugins: [remarkMath, remarkMdxImages, remarkModifiedTime],
        rehypePlugins: [rehypeKatex],
      },
});